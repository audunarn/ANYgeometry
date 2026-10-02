# Intersection continuation evidence

Development evidence for candidate `96308444054a7b002e2897cd3bbf46cf78021338`.
Release artifacts are unchanged. The initial machine-readable snapshot is retained at
`C:/Github/ANYgeometry/reports/intersection-continuation-evidence-9630844.json`.

```json
{
  "recorded_at": "2026-10-02T09:21:42.899482+00:00",
  "candidate_commit": "96308444054a7b002e2897cd3bbf46cf78021338",
  "base_commit": "462ca593cec9bf36106b7baa72efe2587887625d",
  "pull_request": "https://github.com/audunarn/ANYgeometry/pull/12",
  "suites": {
    "kernel": {
      "tests": "1676",
      "failures": "0",
      "errors": "0",
      "skipped": "0",
      "time": "407.760"
    },
    "installed_fixtures": {
      "tests": "6",
      "failures": "0",
      "errors": "0",
      "skipped": "0",
      "time": "16.804"
    }
  },
  "installed_identity": {
    "source_commit": "96308444054a7b002e2897cd3bbf46cf78021338",
    "wheel_sha256": "0dcc5c2585abf1e2e26b5844f4c440af7b0ed746fdcc155478424e1586f4846c",
    "origin": "C:\\Users\\AudunArnesenNyhus\\AppData\\Local\\Temp\\anygeometry-continuation-9630844\\Lib\\site-packages\\anygeometry\\__init__.py",
    "python": "3.13.9 (tags/v3.13.9:8183fa5, Oct 14 2025, 14:09:13) [MSC v.1944 64 bit (AMD64)]",
    "version": "0.4.5",
    "verified_package_files": 75,
    "dependencies": {
      "numpy": "2.5.3",
      "shapely": "2.1.2",
      "pytest": "9.1.1"
    }
  },
  "mesher_commit": "847d6603288fb82a68f2b9ee9aec09eba48b551d",
  "mesher_owner_report": {
    "focused_passed": 26,
    "projection_case": "face 13 at h=0.75: 110 nodes, 19 Q8, 4 T6, CERTIFIED_POSITIVE",
    "unaccepted": "face 11: 7-degree T3 corner under unchanged 15-degree policy",
    "evidence": "C:/Github/ANYmesh/reports/geometry-consumer/probe-branch-wheel-96308444.log",
    "diagnosis": "No authored acute corner: minimum source tangent angle 119.22 degrees. A 0.049578 m side at h=0.75 caused mapped seeding to propagate 56 divisions onto edge 31 (direct demand 4.57); existing automatic fallback lacks strict high-order certification.",
    "status": "Mesher owner implementing bounded seed/placement correction; no acceptance claim for face 11."
  },
  "hosted_development": {
    "run_id": 36989128888,
    "status": "in_progress_at_recording"
  },
  "release": "No publication; existing 0.4.5 release artifacts unchanged",
  "anyfem": "Concrete candidate and fixtures handed off; user authorization requested for separate worker validation/fixes after earlier automatic review blocked implementation delegation.",
  "evidence_files": [
    {
      "path": "C:\\Github\\ANYgeometry\\reports\\intersection-continuation-kernel.log",
      "sha256": "c01f55ab94d6e5dc180751b0d36df1daa2439cd663f364d1b22afdc7d9979117"
    },
    {
      "path": "C:\\Github\\ANYgeometry\\reports\\intersection-continuation-kernel.xml",
      "sha256": "5edc60cac17bdad44a818b044db8f4a0544c2b85ddd4154cef9961b8f9fe6596"
    },
    {
      "path": "C:\\Github\\ANYgeometry\\reports\\intersection-continuation-installed-retry.log",
      "sha256": "fa96fd4f259134572b754529edc52a5a7dc9a7d75a4fa6048c9e64675b57ca1b"
    },
    {
      "path": "C:\\Github\\ANYgeometry\\reports\\intersection-continuation-installed.xml",
      "sha256": "aa70053f9a9e8a157cf9d074c3607bf12810596cff605da32e5cb2dd074461eb"
    },
    {
      "path": "C:\\Github\\ANYgeometry\\reports\\intersection-continuation-installed-identity.json",
      "sha256": "6097ca7afd8fba79aeeb0b257ec36cc8ee3cb50a4e08989b0ba3d6661642f791"
    },
    {
      "path": "C:\\Github\\ANYgeometry\\reports\\intersection-continuation-installed-freeze.txt",
      "sha256": "16555b3c82f8e35ee34b46eb124573689f7c8db16a8582e7ad99cc0fa07c29fc"
    },
    {
      "path": "C:\\Github\\ANYgeometry\\reports\\tangent-wall-events-certified-intervals.log",
      "sha256": "c7cb491c76ce201ffe380876094297d15f1f691faeb8f9fce13272115f5df13c"
    },
    {
      "path": "C:\\Github\\ANYgeometry\\reports\\edge-attachment-remapping-batch-contacts-corrected.log",
      "sha256": "babbd898f0f233917ed1126d4e4ebb370fb935c0ec7ae20e78cb4f75a9c41cf1"
    },
    {
      "path": "C:\\Github\\ANYgeometry\\reports\\consumer-contract-fixture-tests.log",
      "sha256": "e11ba789e4c8fb34e06adc4b890803c39ec6646125e38d8ec0b9e72e2b9cf10b"
    },
    {
      "path": "C:\\Github\\ANYgeometry\\reports\\consumer-contract-9630844\\manifest.json",
      "sha256": "131eca9787177ace623c48011d55eaced348219e68bc804437507c188a0ba7ef"
    },
    {
      "path": "C:\\Github\\ANYgeometry\\reports\\tangent-continuation-baseline-seams.log",
      "sha256": "38cceb5dd3d9249d572d493dbadcfa7f43a698ed1fbbf9ac6bd039ba9165feeb"
    },
    {
      "path": "C:\\Github\\ANYgeometry\\reports\\edge-attachment-remapping-initial.log",
      "sha256": "d12fd6d823b9b62338fad9e43b26ba62cf91423417a90c8119a0b723a2f9fd05"
    },
    {
      "path": "C:\\Github\\ANYgeometry\\reports\\intersection-continuation-install.log",
      "sha256": "84a8ca811f6e268176853b95083eaa2156f30ae351d74819266633d2b834994e"
    },
    {
      "path": "C:\\Github\\ANYgeometry\\reports\\intersection-continuation-installed.log",
      "sha256": "4616de2febd1b72b2c8aa97b5a5c59ce5141420c4be7c48e3374bd078a919019"
    }
  ],
  "review_evidence": [
    {
      "path": "C:\\Users\\AUDUNA~1\\AppData\\Local\\Temp\\anygeometry-wall-review-completeness.log",
      "sha256": "a85284ed4f8c2206067e7078d2e7856f289e4f9c1fb43548437cdcdbd8f5c50b"
    },
    {
      "path": "C:\\Users\\AUDUNA~1\\AppData\\Local\\Temp\\anygeometry-wall-review-scale.log",
      "sha256": "c8f7c645c24b417502c07d595324fe75fe6e1825e55c1aed61ad6719ab42d8a3"
    },
    {
      "path": "C:\\Users\\AUDUNA~1\\AppData\\Local\\Temp\\anygeometry-remap-review-member-references-confirmed.log",
      "sha256": "e15de08d1cab1f6c817d2d40761b48a2ac54344f666926cb73184f4d7be03f99"
    },
    {
      "path": "C:\\Users\\AUDUNA~1\\AppData\\Local\\Temp\\anygeometry-definition-binding-after.log",
      "sha256": "6656ace4e74523708f0b82c74e87badd52ad2e63c34860036edc0a6c0bd3d612"
    },
    {
      "path": "C:\\Users\\AUDUNA~1\\AppData\\Local\\Temp\\anygeometry-branch-projection-polished.log",
      "sha256": "1a8be6cd81813d6f214747b4b7f369893be1c315f9217b78fa6036115b8b8add"
    },
    {
      "path": "C:\\Users\\AUDUNA~1\\AppData\\Local\\Temp\\anygeometry-chart-validation-after.log",
      "sha256": "2a4db6d2155b78ebf2caa83f987757991f8553fa7d7b5f98d0371f7b54144dcd"
    }
  ],
  "independent_review": "OpenAI read-only findings resolved and reconciled; supplementary Mistral tangency review incomplete, not counted as clean acceptance."
}
```
