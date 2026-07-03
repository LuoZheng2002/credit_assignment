use std::{
    collections::BTreeMap,
    fs::File,
    io::{BufRead, BufReader, Cursor, Seek, SeekFrom},
    path::{Path, PathBuf},
};

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_jsonlines::BufReadExt;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct HybridSftDatasetEntry {
    pub flat_id: usize,
    pub dataset_name: String,
    pub question_id: usize,
    pub question: String,
    pub correct_answer: String,
    pub prompt: String,
    pub reference_trajectory: String,
}

pub struct HybridSftDatasetStore {
    file_path: PathBuf,
    line_offsets: Vec<u64>,
    cache: Mutex<BTreeMap<usize, HybridSftDatasetEntry>>,
}

pub struct HybridSftDatasetIter {
    inner: Box<dyn Iterator<Item = Result<HybridSftDatasetEntry, std::io::Error>> + Send>,
    index: usize,
}

impl Iterator for HybridSftDatasetIter {
    type Item = Result<(usize, HybridSftDatasetEntry), String>;

    fn next(&mut self) -> Option<Self::Item> {
        let entry = match self.inner.next()? {
            Ok(e) => e,
            Err(err) => {
                return Some(Err(format!(
                    "Failed to deserialize JSONL record at index {}: {}",
                    self.index, err
                )));
            }
        };
        let idx = self.index;
        self.index += 1;
        Some(Ok((idx, entry)))
    }
}

impl HybridSftDatasetStore {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, String> {
        let file_path = path.as_ref().to_path_buf();
        let file = File::open(&file_path).map_err(|err| {
            format!(
                "Failed to open hybrid SFT dataset JSONL file {}: {}",
                file_path.display(),
                err
            )
        })?;
        let mut reader = BufReader::new(file);
        let mut line = String::new();
        let mut line_offsets = Vec::new();
        let mut offset = 0u64;

        loop {
            line.clear();
            let bytes_read = reader.read_line(&mut line).map_err(|err| {
                format!(
                    "Failed to read hybrid SFT dataset JSONL file {} while indexing line offsets: {}",
                    file_path.display(),
                    err
                )
            })?;
            if bytes_read == 0 {
                break;
            }
            line_offsets.push(offset);
            offset += bytes_read as u64;
        }

        Ok(Self {
            file_path,
            line_offsets,
            cache: Mutex::new(BTreeMap::new()),
        })
    }

    pub fn len(&self) -> usize {
        self.line_offsets.len()
    }

    pub fn iter(&self) -> Result<HybridSftDatasetIter, String> {
        let file = File::open(&self.file_path).map_err(|err| {
            format!(
                "Failed to open hybrid SFT dataset JSONL file {} for iteration: {}",
                self.file_path.display(),
                err
            )
        })?;
        let reader = BufReader::new(file);
        let inner = reader.json_lines::<HybridSftDatasetEntry>();
        Ok(HybridSftDatasetIter {
            inner: Box::new(inner),
            index: 0,
        })
    }

    pub fn get(&self, key: usize) -> Result<Option<HybridSftDatasetEntry>, String> {
        if let Some(entry) = self.cache.lock().get(&key).cloned() {
            return Ok(Some(entry));
        }

        let Some(offset) = self.line_offsets.get(key).copied() else {
            return Ok(None);
        };

        let mut file = File::open(&self.file_path).map_err(|err| {
            format!(
                "Failed to open hybrid SFT dataset JSONL file {} for lookup of key {}: {}",
                self.file_path.display(),
                key,
                err
            )
        })?;
        file.seek(SeekFrom::Start(offset)).map_err(|err| {
            format!(
                "Failed to seek hybrid SFT dataset JSONL file {} to offset {} for key {}: {}",
                self.file_path.display(),
                offset,
                key,
                err
            )
        })?;

        let mut line = String::new();
        let mut reader = BufReader::new(file);
        let bytes_read = reader.read_line(&mut line).map_err(|err| {
            format!(
                "Failed to read line for key {} from hybrid SFT dataset JSONL file {}: {}",
                key,
                self.file_path.display(),
                err
            )
        })?;
        if bytes_read == 0 {
            return Err(format!(
                "Reached EOF while loading key {} from hybrid SFT dataset JSONL file {}",
                key,
                self.file_path.display()
            ));
        }

        let mut items =
            BufReader::new(Cursor::new(line.into_bytes())).json_lines::<HybridSftDatasetEntry>();
        let entry = items
            .next()
            .transpose()
            .map_err(|err| {
                format!(
                    "Failed to deserialize key {} from hybrid SFT dataset JSONL file {}: {}",
                    key,
                    self.file_path.display(),
                    err
                )
            })?
            .ok_or_else(|| {
                format!(
                    "Missing JSON value for key {} in hybrid SFT dataset JSONL file {}",
                    key,
                    self.file_path.display()
                )
            })?;

        self.cache.lock().insert(key, entry.clone());
        Ok(Some(entry))
    }
}

pub fn hybrid_sft_dataset_file_path() -> String {
    "datasets/hybrid_sft_deepseek_v4_pro.jsonl".to_string()
}

pub fn open_hybrid_sft_dataset() -> HybridSftDatasetStore {
    HybridSftDatasetStore::open(hybrid_sft_dataset_file_path()).unwrap_or_else(|e| {
        panic!(
            "Failed to open hybrid SFT dataset JSONL store at {}: {}",
            hybrid_sft_dataset_file_path(),
            e
        )
    })
}
