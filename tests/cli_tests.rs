use assert_cmd::Command;
use predicates::prelude::*;
use serde_json::Value;
use tempfile::tempdir;

const SAMPLE_LGBM_MODEL: &str = r#"tree
version=v4
num_class=1
num_tree_per_iteration=1
label_index=0
max_feature_idx=2
objective=binary sigmoid:1
feature_names=gpa miss_rate program
feature_infos=[1.5:4.0] [0.0:1.0] -1:0:1:2

Tree=0
num_leaves=3
num_cat=1
split_feature=1 2
split_gain=80.0 25.0
threshold=0.5 0
decision_type=2 1
left_child=1 -3
right_child=-1 -2
leaf_value=-1.0 0.5 1.5
leaf_count=100 60 40
internal_count=200 100
cat_boundaries=0 1
cat_threshold=1
is_linear=0
shrinkage=1

end of trees

pandas_categorical:[["Cert", "MDiv", "MAC"]]
"#;

#[test]
fn inspect_and_score_with_exact_treeshap_efficiency_axiom() {
    let dir = tempdir().unwrap();
    let model_path = dir.path().join("model.txt");
    let data_path = dir.path().join("students.csv");
    let html_path = dir.path().join("report.html");
    std::fs::write(&model_path, SAMPLE_LGBM_MODEL).unwrap();
    std::fs::write(
        &data_path,
        "id,gpa,miss_rate,program\nS1,3.8,0.80,MDiv\nS2,3.5,0.10,Cert\nS3,2.2,0.10,MAC\n",
    )
    .unwrap();

    // 1. Inspect
    let insp = Command::cargo_bin("scorekeel")
        .unwrap()
        .args(["inspect", model_path.to_str().unwrap(), "--format", "agent"])
        .output()
        .unwrap();
    assert!(insp.status.success());
    let ij: Value = serde_json::from_slice(&insp.stdout).unwrap();
    assert_eq!(ij["num_trees"], 1);
    assert_eq!(ij["num_features"], 3);

    // 2. Score with exact TreeSHAP + verify Shapley efficiency axiom:
    //    expected_value + sum(shap_values) == raw_score!
    let sc = Command::cargo_bin("scorekeel")
        .unwrap()
        .args([
            "score",
            model_path.to_str().unwrap(),
            data_path.to_str().unwrap(),
            "--shap",
            "--id-col",
            "id",
            "--html",
            html_path.to_str().unwrap(),
            "--format",
            "agent",
        ])
        .output()
        .unwrap();
    assert!(sc.status.success());
    assert!(html_path.exists());
    let sj: Value = serde_json::from_slice(&sc.stdout).unwrap();
    let preds = sj["predictions"].as_array().unwrap();
    assert_eq!(preds.len(), 3);

    // S1: miss_rate=0.80 > 0.5 -> right_child=-1 -> leaf 0 = -1.0
    assert!((preds[0]["raw_score"].as_f64().unwrap() - (-1.0)).abs() < 1e-5);
    // S2: miss_rate=0.10 <= 0.5 -> node 1 (cat bitset=1 -> bit 0 = "Cert") -> left=-3 -> leaf 2 = 1.5
    assert!((preds[1]["raw_score"].as_f64().unwrap() - 1.5).abs() < 1e-5);
    // S3: miss_rate=0.10 <= 0.5 -> "MAC" (code 2 not in bitset 1) -> right=-2 -> leaf 1 = 0.5
    assert!((preds[2]["raw_score"].as_f64().unwrap() - 0.5).abs() < 1e-5);

    // Exact Shapley efficiency axiom on EVERY row!
    for row in preds {
        let raw = row["raw_score"].as_f64().unwrap();
        let ev = row["expected_value"].as_f64().unwrap();
        let shap_sum: f64 = row["shap_values"]
            .as_object()
            .unwrap()
            .values()
            .map(|v| v.as_f64().unwrap())
            .sum();
        assert!(
            ((ev + shap_sum) - raw).abs() < 1e-4,
            "efficiency axiom failed: ev={ev} + shap_sum={shap_sum} != raw={raw}"
        );
    }
}

#[test]
fn skill_prints_and_installs() {
    Command::cargo_bin("scorekeel")
        .unwrap()
        .arg("skill")
        .assert()
        .success()
        .stdout(predicate::str::contains("name: scorekeel"));
}
