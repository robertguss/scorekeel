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
fn rejects_malformed_cyclic_and_overflow_models_and_handles_empty_html_and_covers() {
    let dir = tempdir().unwrap();

    // 1. Cyclic tree + threshold=1e30 rejected cleanly (no hang / stack overflow / wraparound panic)
    let bad_model = dir.path().join("cyclic.txt");
    std::fs::write(
        &bad_model,
        "tree\nnum_class=1\nfeature_names=x\nTree=0\nnum_leaves=2\nsplit_feature=0\nthreshold=1e30\ndecision_type=1\nleft_child=0\nright_child=-1\nleaf_value=0.5 1.0\ncat_boundaries=0 1\ncat_threshold=1\n",
    )
    .unwrap();
    Command::cargo_bin("scorekeel")
        .unwrap()
        .args(["inspect", bad_model.to_str().unwrap()])
        .assert()
        .code(2);

    // 2. is_linear=1 rejected cleanly
    let lin_model = dir.path().join("linear.txt");
    std::fs::write(
        &lin_model,
        "tree\nnum_class=1\nfeature_names=x\nTree=0\nnum_leaves=1\nleaf_value=0.5\nis_linear=1\n",
    )
    .unwrap();
    Command::cargo_bin("scorekeel")
        .unwrap()
        .args(["inspect", lin_model.to_str().unwrap()])
        .assert()
        .code(2);

    // 3. Empty CSV + --html succeeds without panic; duplicate CSV headers rejected
    let model_path = dir.path().join("model.txt");
    std::fs::write(&model_path, SAMPLE_LGBM_MODEL).unwrap();
    let empty_csv = dir.path().join("empty.csv");
    std::fs::write(&empty_csv, "id,gpa,miss_rate,program\n").unwrap();
    let empty_html = dir.path().join("empty.html");
    Command::cargo_bin("scorekeel")
        .unwrap()
        .args([
            "score",
            model_path.to_str().unwrap(),
            empty_csv.to_str().unwrap(),
            "--html",
            empty_html.to_str().unwrap(),
            "--format",
            "agent",
        ])
        .assert()
        .success();
    assert!(empty_html.exists());

    let dup_csv = dir.path().join("dup.csv");
    std::fs::write(&dup_csv, "gpa,gpa\n3.5,2.0\n").unwrap();
    Command::cargo_bin("scorekeel")
        .unwrap()
        .args(["score", model_path.to_str().unwrap(), dup_csv.to_str().unwrap()])
        .assert()
        .code(2);

    // 4. --platt NaN,0 rejected
    Command::cargo_bin("scorekeel")
        .unwrap()
        .args([
            "score",
            model_path.to_str().unwrap(),
            empty_csv.to_str().unwrap(),
            "--platt",
            "NaN,0",
        ])
        .assert()
        .code(2);

    // 5. TreeSHAP without leaf_count/internal_count still satisfies exact additivity axiom!
    let nocount_model = dir.path().join("nocount.txt");
    let stripped: String = SAMPLE_LGBM_MODEL
        .lines()
        .filter(|l| !l.starts_with("leaf_count=") && !l.starts_with("internal_count="))
        .collect::<Vec<_>>()
        .join("\n");
    std::fs::write(&nocount_model, stripped).unwrap();
    let data_path = dir.path().join("rows.csv");
    std::fs::write(
        &data_path,
        "id,gpa,miss_rate,program\nS2,3.5,0.10,Cert\n",
    )
    .unwrap();
    let sc = Command::cargo_bin("scorekeel")
        .unwrap()
        .args([
            "score",
            nocount_model.to_str().unwrap(),
            data_path.to_str().unwrap(),
            "--shap",
            "--format",
            "agent",
        ])
        .output()
        .unwrap();
    assert!(sc.status.success());
    let sj: Value = serde_json::from_slice(&sc.stdout).unwrap();
    let row = &sj["predictions"][0];
    let raw = row["raw_score"].as_f64().unwrap();
    let ev = row["expected_value"].as_f64().unwrap();
    let shap_sum: f64 = row["shap_values"]
        .as_object()
        .unwrap()
        .values()
        .map(|v| v.as_f64().unwrap())
        .sum();
    assert!(((ev + shap_sum) - raw).abs() < 1e-4, "ev={ev} + sum={shap_sum} != raw={raw}");

    // 6. Anchor=0 with raw=0 produces 0.0 (not 1.0) + shifted pandas_categorical with constant col 0
    let reg_model = dir.path().join("reg_cat.txt");
    std::fs::write(
        &reg_model,
        r#"tree
version=v4
num_class=1
objective=regression
feature_names=const_cat active_cat
feature_infos=none -1:0:1

Tree=0
num_leaves=2
split_feature=1
threshold=0
decision_type=1
left_child=-1
right_child=-2
leaf_value=0.0 1.0
cat_boundaries=0 1
cat_threshold=1

end of trees

parameters:
[categorical_feature: 0,1]
end of parameters

pandas_categorical:[["U1"], ["X", "Y"]]
"#,
    )
    .unwrap();
    let reg_csv = dir.path().join("reg.csv");
    std::fs::write(&reg_csv, "const_cat,active_cat,prior\nU1,X,0\nU1,Y,0\n").unwrap();
    let rsc = Command::cargo_bin("scorekeel")
        .unwrap()
        .args([
            "score",
            reg_model.to_str().unwrap(),
            reg_csv.to_str().unwrap(),
            "--anchor-col",
            "prior",
            "--format",
            "agent",
        ])
        .output()
        .unwrap();
    assert!(rsc.status.success());
    let rj: Value = serde_json::from_slice(&rsc.stdout).unwrap();
    // Row 0 ("X", code 0 in bitset 1 -> leaf 0 = 0.0, anchor=0 -> (0+1)*exp(0)-1 = 0.0!)
    assert!((rj["predictions"][0]["prediction"].as_f64().unwrap() - 0.0).abs() < 1e-5);
    // Row 1 ("Y", code 1 not in bitset 1 -> leaf 1 = 1.0, anchor=0 -> (0+1)*exp(1)-1 = 1.718282!)
    assert!((rj["predictions"][1]["prediction"].as_f64().unwrap() - (1.0_f64.exp() - 1.0)).abs() < 1e-4);
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
