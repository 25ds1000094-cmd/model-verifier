from flask import Flask, request, jsonify
import hashlib
import json
import math
import re
import os

app = Flask(__name__)

REQUIRED_FILES = [
    "README.md",
    "training_manifest.json",
    "evaluation.json",
    "inventory.json",
    "adapter_model.safetensors",
    "adapter_config.json",
]

UNSAFE_EXTENSIONS = (
    ".bin",
    ".pt",
    ".pth",
    ".pkl",
    ".pickle",
)

MAX_SAFE_INTEGER = 9007199254740991


def utf8_bytes(text):
    return text.encode("utf-8")


def sha256_text(text):
    return hashlib.sha256(utf8_bytes(text)).hexdigest()


def compact_json(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":")
    )


def add_violation(violations, code):
    violations.add(code)


def non_empty_string(value):
    return isinstance(value, str) and len(value) > 0


def safe_integer(value):
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 1 <= value <= MAX_SAFE_INTEGER
    )


def parse_json_file(files, filename, violations):
    if filename not in files:
        return None

    try:
        return json.loads(files[filename])
    except (json.JSONDecodeError, UnicodeEncodeError):
        add_violation(violations, f"INVALID_JSON:{filename}")
        return None


def validate_policy(policy, violations):
    if not isinstance(policy, dict):
        add_violation(violations, "INVALID_POLICY")
        return

    required_slices = policy.get("requiredSlices")

    if (
        not isinstance(required_slices, list)
        or len(required_slices) == 0
        or any(not non_empty_string(x) for x in required_slices)
        or len(set(required_slices)) != len(required_slices)
    ):
        add_violation(violations, "INVALID_POLICY")

    for field in ["license", "intendedUse", "limitations"]:
        if not non_empty_string(policy.get(field)):
            add_violation(violations, "INVALID_POLICY")


def validate_files(files, violations):
    if not isinstance(files, dict):
        add_violation(violations, "INVALID_POLICY")
        return

    for filename, content in files.items():
        if not isinstance(filename, str) or not isinstance(content, str):
            add_violation(violations, "INVALID_POLICY")


def check_required_files(files, violations):
    for filename in REQUIRED_FILES:
        if filename not in files:
            add_violation(
                violations,
                f"MISSING_FILE:{filename}"
            )


def check_unsafe_weights(files, violations):
    for filename in files:
        if filename.lower().endswith(UNSAFE_EXTENSIONS):
            add_violation(violations, "UNSAFE_WEIGHTS")


def verify_inventory(files, inventory, violations):
    if not isinstance(inventory, list):
        add_violation(violations, "INVENTORY_MISMATCH")
        return ""

    # Every file except inventory.json must appear.
    actual_names = [
        filename
        for filename in files
        if filename != "inventory.json"
    ]

    # Sort using UTF-8 bytes, not normal locale sorting.
    actual_names.sort(key=lambda x: x.encode("utf-8"))

    expected_inventory = []

    for filename in actual_names:
        raw = utf8_bytes(files[filename])

        expected_inventory.append({
            "name": filename,
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest()
        })

    # Check exact key order.
    for entry in inventory:
        if not isinstance(entry, dict):
            add_violation(violations, "INVENTORY_MISMATCH")
            continue

        if list(entry.keys()) != [
            "name",
            "bytes",
            "sha256"
        ]:
            add_violation(violations, "INVENTORY_MISMATCH")

    # Check exact values and ordering.
    if inventory != expected_inventory:
        add_violation(violations, "INVENTORY_MISMATCH")

    # Check that no actual file was left out.
    inventory_names = set()

    for entry in inventory:
        if isinstance(entry, dict):
            name = entry.get("name")

            if isinstance(name, str):
                inventory_names.add(name)

    for filename in actual_names:
        if filename not in inventory_names:
            add_violation(violations, "UNTRACKED_FILE")

    # Important:
    # Hash the recomputed inventory, not the attacker's inventory.
    return sha256_text(compact_json(expected_inventory))


def verify_adapter_config(config, violations):
    if not isinstance(config, dict):
        add_violation(
            violations,
            "INVALID_ADAPTER_CONFIG"
        )
        return

    r = config.get("r")
    target_modules = config.get("target_modules")

    if not safe_integer(r):
        add_violation(
            violations,
            "INVALID_ADAPTER_CONFIG"
        )

    if (
        not isinstance(target_modules, list)
        or len(target_modules) == 0
        or any(not non_empty_string(x) for x in target_modules)
        or len(set(target_modules)) != len(target_modules)
    ):
        add_violation(
            violations,
            "INVALID_ADAPTER_CONFIG"
        )


def verify_training_manifest(
    manifest,
    files,
    violations
):
    if not isinstance(manifest, dict):
        add_violation(
            violations,
            "INVALID_TRAINING_MANIFEST"
        )
        return

    required_fields = [
        "baseRevision",
        "task",
        "datasetDigest",
        "codeDigest",
        "trainingConfigDigest",
        "modelArtifactDigest",
        "evaluationArtifactDigest",
    ]

    for field in required_fields:
        if field not in manifest:
            add_violation(
                violations,
                f"MISSING_MANIFEST_FIELD:{field}"
            )
        elif not non_empty_string(manifest[field]):
            add_violation(
                violations,
                "INVALID_TRAINING_MANIFEST"
            )

    base_revision = manifest.get("baseRevision")

    if (
        not isinstance(base_revision, str)
        or re.fullmatch(
            r"[0-9a-f]{40}",
            base_revision
        ) is None
    ):
        add_violation(
            violations,
            "MUTABLE_BASE_REVISION"
        )

    # Check model artifact digest.
    if "adapter_model.safetensors" in files:
        actual_model_digest = sha256_text(
            files["adapter_model.safetensors"]
        )

        if manifest.get("modelArtifactDigest") != actual_model_digest:
            add_violation(
                violations,
                "MODEL_ARTIFACT_MISMATCH"
            )

    # Check evaluation artifact digest.
    if "evaluation.json" in files:
        actual_evaluation_digest = sha256_text(
            files["evaluation.json"]
        )

        if (
            manifest.get("evaluationArtifactDigest")
            != actual_evaluation_digest
        ):
            add_violation(
                violations,
                "EVALUATION_ARTIFACT_MISMATCH"
            )


def verify_evaluation(
    evaluation,
    manifest,
    policy,
    violations
):
    if not isinstance(evaluation, dict):
        add_violation(
            violations,
            "INVALID_EVALUATION"
        )
        return

    # Evaluation must refer to the same model.
    if (
        evaluation.get("modelArtifactDigest")
        != manifest.get("modelArtifactDigest")
    ):
        add_violation(
            violations,
            "EVALUATION_DIGEST_MISMATCH"
        )

    # Aggregate must be a finite number between 0 and 1.
    aggregate = evaluation.get("aggregate")

    if (
        not isinstance(aggregate, (int, float))
        or isinstance(aggregate, bool)
        or not math.isfinite(aggregate)
        or aggregate < 0
        or aggregate > 1
    ):
        add_violation(
            violations,
            "INVALID_AGGREGATE"
        )

    slices = evaluation.get("slices")

    if not isinstance(slices, dict):
        for required in policy.get("requiredSlices", []):
            add_violation(
                violations,
                f"MISSING_SLICE:{required}"
            )
        return

    for required in policy.get("requiredSlices", []):
        if required not in slices:
            add_violation(
                violations,
                f"MISSING_SLICE:{required}"
            )
            continue

        value = slices[required]

        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value < 0
            or value > 1
        ):
            add_violation(
                violations,
                f"SLICE_RANGE:{required}"
            )


MODEL_CARD_PREFIX = "<!-- tds-model-card "


def find_model_cards(readme):
    positions = []
    start = 0

    while True:
        position = readme.find(
            MODEL_CARD_PREFIX,
            start
        )

        if position == -1:
            break

        positions.append(position)
        start = position + len(MODEL_CARD_PREFIX)

    return positions


def verify_model_card(
    readme,
    manifest,
    policy,
    violations
):
    positions = find_model_cards(readme)

    # No markers.
    if len(positions) == 0:
        add_violation(
            violations,
            "MODEL_CARD_COUNT"
        )
        add_violation(
            violations,
            "MISSING_MODEL_CARD"
        )
        return

    # More than one marker.
    if len(positions) > 1:
        add_violation(
            violations,
            "MODEL_CARD_COUNT"
        )
        return

    # Exactly one marker.
    payload_start = (
        positions[0] + len(MODEL_CARD_PREFIX)
    )

    payload_end = readme.find(
        "-->",
        payload_start
    )

    if payload_end == -1:
        add_violation(
            violations,
            "INVALID_MODEL_CARD"
        )
        return

    payload = readme[
        payload_start:payload_end
    ].strip()

    try:
        card = json.loads(payload)
    except json.JSONDecodeError:
        add_violation(
            violations,
            "INVALID_MODEL_CARD"
        )
        return

    if not isinstance(card, dict):
        add_violation(
            violations,
            "INVALID_MODEL_CARD"
        )
        return

    expected = {
        "task": manifest.get("task"),
        "baseRevision": manifest.get("baseRevision"),
        "datasetDigest": manifest.get("datasetDigest"),
        "modelArtifactDigest": manifest.get(
            "modelArtifactDigest"
        ),
        "license": policy.get("license"),
        "intendedUse": policy.get("intendedUse"),
        "limitations": policy.get("limitations"),
    }

    for field, expected_value in expected.items():
        if card.get(field) != expected_value:
            add_violation(
                violations,
                "MODEL_CARD_MISMATCH"
            )
            return


@app.post("/verify-bundle")
def verify_bundle():

    # -------------------------------------------------
    # 1. Check the incoming request.
    # -------------------------------------------------

    if not request.is_json:
        return jsonify({
            "error": "INVALID_INPUT"
        }), 400

    body = request.get_json(silent=True)

    if not isinstance(body, dict):
        return jsonify({
            "error": "INVALID_INPUT"
        }), 400

    # Missing policy = HTTP 400.
    if "policy" not in body:
        return jsonify({
            "error": "INVALID_INPUT"
        }), 400

    # files must be an object.
    files = body.get("files")

    if not isinstance(files, dict):
        return jsonify({
            "error": "INVALID_INPUT"
        }), 400

    policy = body.get("policy")

    # -------------------------------------------------
    # 2. Start collecting violations.
    # -------------------------------------------------

    violations = set()

    validate_policy(
        policy,
        violations
    )

    validate_files(
        files,
        violations
    )

    check_required_files(
        files,
        violations
    )

    check_unsafe_weights(
        files,
        violations
    )

    # -------------------------------------------------
    # 3. Parse JSON files.
    # -------------------------------------------------

    inventory = parse_json_file(
        files,
        "inventory.json",
        violations
    )

    adapter_config = parse_json_file(
        files,
        "adapter_config.json",
        violations
    )

    manifest = parse_json_file(
        files,
        "training_manifest.json",
        violations
    )

    evaluation = parse_json_file(
        files,
        "evaluation.json",
        violations
    )

    # -------------------------------------------------
    # 4. Verify inventory.
    # -------------------------------------------------

    inventory_digest = ""

    if "inventory.json" in files:
        inventory_digest = verify_inventory(
            files,
            inventory,
            violations
        )

    # -------------------------------------------------
    # 5. Verify adapter config.
    # -------------------------------------------------

    if "adapter_config.json" in files:
        verify_adapter_config(
            adapter_config,
            violations
        )

    # -------------------------------------------------
    # 6. Verify training manifest.
    # -------------------------------------------------

    if "training_manifest.json" in files:
        verify_training_manifest(
            manifest,
            files,
            violations
        )

    # -------------------------------------------------
    # 7. Verify evaluation.
    # -------------------------------------------------

    if (
        "evaluation.json" in files
        and "training_manifest.json" in files
        and isinstance(manifest, dict)
    ):
        verify_evaluation(
            evaluation,
            manifest,
            policy,
            violations
        )

    # -------------------------------------------------
    # 8. Verify README model card.
    # -------------------------------------------------

    if "README.md" in files:
        if isinstance(manifest, dict):
            verify_model_card(
                files["README.md"],
                manifest,
                policy,
                violations
            )
        else:
            positions = find_model_cards(
                files["README.md"]
            )

            if len(positions) == 0:
                add_violation(
                    violations,
                    "MODEL_CARD_COUNT"
                )
                add_violation(
                    violations,
                    "MISSING_MODEL_CARD"
                )
            elif len(positions) > 1:
                add_violation(
                    violations,
                    "MODEL_CARD_COUNT"
                )
            else:
                add_violation(
                    violations,
                    "MODEL_CARD_MISMATCH"
                )

    # -------------------------------------------------
    # 9. Remove duplicates and sort by UTF-8 bytes.
    # -------------------------------------------------

    sorted_violations = sorted(
        violations,
        key=lambda x: x.encode("utf-8")
    )

    decision = (
        "admit"
        if len(sorted_violations) == 0
        else "reject"
    )

    # -------------------------------------------------
    # 10. Return EXACT required response shape.
    # -------------------------------------------------

    return jsonify({
        "decision": decision,
        "violations": sorted_violations,
        "inventoryDigest": inventory_digest
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))

    app.run(
        host="0.0.0.0",
        port=port
    )
