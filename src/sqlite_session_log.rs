use crate::agent::tree_action::TreeAction;
use rusqlite::{Connection, OptionalExtension, Row, params};
use std::path::PathBuf;

#[derive(Debug)]
pub struct SqliteSessionLogStore {
    db_path: PathBuf,
    connection: Connection,
}

impl SqliteSessionLogStore {
    pub fn new(db_path: impl Into<PathBuf>) -> Result<Self, String> {
        let db_path = db_path.into();
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                format!(
                    "Failed to create parent directory for sqlite session log database {}: {}",
                    db_path.display(),
                    e
                )
            })?;
        }
        let connection = Connection::open(&db_path).map_err(|e| {
            format!(
                "Failed to open sqlite session log database {}: {}",
                db_path.display(),
                e
            )
        })?;
        Ok(Self {
            db_path,
            connection,
        })
    }

    pub fn append_action(&self, question_id: usize, action: &TreeAction) -> Result<(), String> {
        self.initialize_question_table(question_id)?;
        let table_name = Self::question_table_name(question_id);
        let next_index: i64 = self
            .connection
            .query_row(&format!("SELECT COUNT(*) FROM {}", table_name), [], |row| {
                row.get(0)
            })
            .map_err(|e| {
                format!(
                    "Failed to count existing session log actions in table {} at {}: {}",
                    table_name,
                    self.db_path.display(),
                    e
                )
            })?;
        assert!(next_index >= 0, "sqlite COUNT(*) must be non-negative");
        let payload_json = serde_json::to_string(action).map_err(|e| {
            format!(
                "Failed to serialize session log action for table {} in {}: {}",
                table_name,
                self.db_path.display(),
                e
            )
        })?;
        self.connection
            .execute(
                &format!(
                    "
                    INSERT INTO {} (id, payload_json)
                    VALUES (?1, ?2)
                    ",
                    table_name
                ),
                params![next_index, payload_json],
            )
            .map_err(|e| {
                format!(
                    "Failed to append session log action at index {} in table {} at {}: {}",
                    next_index,
                    table_name,
                    self.db_path.display(),
                    e
                )
            })?;
        Ok(())
    }

    pub fn load_question_actions(&self, question_id: usize) -> Result<Vec<TreeAction>, String> {
        if !self.question_table_exists(question_id)? {
            return Ok(Vec::new());
        }
        let table_name = Self::question_table_name(question_id);
        let mut statement = self
            .connection
            .prepare(&format!(
                "
                SELECT payload_json
                FROM {}
                ORDER BY id ASC
                ",
                table_name
            ))
            .map_err(|e| {
                format!(
                    "Failed to prepare ordered session log scan statement for table {} in {}: {}",
                    table_name,
                    self.db_path.display(),
                    e
                )
            })?;
        let rows = statement.query_map([], decode_payload_row).map_err(|e| {
            format!(
                "Failed to execute ordered session log scan query for table {} in {}: {}",
                table_name,
                self.db_path.display(),
                e
            )
        })?;
        let mut actions = Vec::new();
        for row in rows {
            actions.push(row.map_err(|e| {
                format!(
                    "Failed to read session log row from table {} in {}: {}",
                    table_name,
                    self.db_path.display(),
                    e
                )
            })?);
        }
        Ok(actions)
    }

    pub fn drop_question_table(&self, question_id: usize) -> Result<(), String> {
        let table_name = Self::question_table_name(question_id);
        self.connection
            .execute(&format!("DROP TABLE IF EXISTS {}", table_name), [])
            .map_err(|e| {
                format!(
                    "Failed to drop session log table {} in {}: {}",
                    table_name,
                    self.db_path.display(),
                    e
                )
            })?;
        Ok(())
    }

    fn initialize_question_table(&self, question_id: usize) -> Result<(), String> {
        let table_name = Self::question_table_name(question_id);
        self.connection
            .execute_batch(&format!(
                "
                CREATE TABLE IF NOT EXISTS {} (
                    id INTEGER PRIMARY KEY,
                    payload_json TEXT NOT NULL
                );
                ",
                table_name
            ))
            .map_err(|e| {
                format!(
                    "Failed to initialize session log table {} in {}: {}",
                    table_name,
                    self.db_path.display(),
                    e
                )
            })?;
        Ok(())
    }

    fn question_table_exists(&self, question_id: usize) -> Result<bool, String> {
        let table_name = Self::question_table_name(question_id);
        let existing_table_name: Option<String> = self
            .connection
            .query_row(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?1",
                params![table_name],
                |row| row.get(0),
            )
            .optional()
            .map_err(|e| {
                format!(
                    "Failed to query sqlite_master for session log table {} in {}: {}",
                    table_name,
                    self.db_path.display(),
                    e
                )
            })?;
        Ok(existing_table_name.is_some())
    }

    fn question_table_name(question_id: usize) -> String {
        format!("question_{}", question_id)
    }
}

fn decode_payload_row(row: &Row<'_>) -> rusqlite::Result<TreeAction> {
    let payload_json: String = row.get(0)?;
    serde_json::from_str(&payload_json).map_err(|e| {
        rusqlite::Error::FromSqlConversionFailure(0, rusqlite::types::Type::Text, Box::new(e))
    })
}
