use credit_assignment::deepmath::generate_raw_answers::Model;
use credit_assignment::multi_agent::rollout::rollout;
use pyo3::Python;
use rand::{SeedableRng, rngs::StdRng};

#[tokio::main]
async fn main() {
    Python::initialize();
    dotenvy::dotenv().ok();
    let question = "Let $(x_1,y_1),$ $(x_2,y_2),$ $\\dots,$ \
$(x_n,y_n)$ be the solutions to\n\\begin{align*}\n|x - 3| &= |y - 9|, \\\\\n|x - 9| &= 2|y - 3|.\n\\end{align*}Find $x_1 + y_1 + x_2 + y_2 + \\dots + x_n + y_n.$".to_string();
    // let question = "Find the sum of all prime numbers less than 10000.".to_string();
    let client = reqwest::Client::new();
    let mut rng = StdRng::seed_from_u64(42); // Use a fixed seed for reproducibility
    let verifier_probability = 1.0; // Set your desired verifier probability
    rollout(
        0,
        question,
        client,
        Model::Gpt4o,
        verifier_probability,
        &mut rng,
    )
    .await;
}
