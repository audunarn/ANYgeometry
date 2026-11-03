# Native-platform test contracts

The initial platform migration preserved every Windows body exactly. The subsequent explicit owner-approved withdrawn-512MiB delta changes only the two resource functions named below: every original success assertion/limit remains, while a measured resource terminal can be accepted only for reuse=False. Seven files and the other25 native Windows bodies remain exact; the portable empty-plan body also remains unchanged. No native ownership assertion, cancellation expectation, numerical tolerance, production code, guard or pin is relaxed. Nine original files are retained under .pp1-evidence/r10-platform-contract/originals with SHA256 identities in assignment-map.json. verify_body_identity.py compares every original function source segment exactly (including arguments/body text), AST and original decorators, then requires full module AST equivalence after removing only the new helper import/decorator and the old host-memory module skip. Results are body-identity-proof.json.

Every selected test node still executes. tests/_planning_platform_contract.py wraps only native-dependent cases and injects the request fixture without changing original parameters. On os.name==nt it records execution_scope=original-Windows-assertions and invokes the original function. A Windows failure remains a failure: recording scope does not imply qualification or success. On other hosts it records execution_scope=unsupported-platform-refusal, EXECUTES the real OwnedTree(tmp_path), requires the exact QualificationDecline type/text owned planner requires Windows Job containment, and verifies the initially empty directory remains empty. No fake WinAPI, skip, xfail, test selection exemption or broad catch is used. The platform branch does not claim native creation, cancellation after admission, memory enforcement, numerical worker execution or cleanup of a tree that was never created.

The current feature requires Windows Job containment. The platform adapter alone invokes the original function on Windows; that OS condition never established qualification of every Windows numerical runtime. The subsequent explicit numerical-profile matrix below distinguishes the raw resource helper ABI from numerical P/W admission. Current 4GB and private 512MiB/nonreuse restrictions are untouched. A mandatory qualified Windows positive CI lane remains necessary; generic POSIX nodes test the exact unsupported-host gate. Local Windows check evidence must not be generalized to other runtimes.

## Actual CI evidence

The parent-provided linux-ci4.log and mac-ci4.log each report19 failing assigned nodes (their shard covers only a subset). Direct native cases fail at OwnedTree Windows-only containment; stage/session/cancellation positives can decline earlier and thus never reach their old worker/terminal assertions. Both logs are preserved under .pp1-evidence/r9-delivery-ci. No missing CI results are inferred as passes.

## Per-function mapping

Each row covers every existing parameter variant. Windows function text is retained exactly except the two explicitly approved resource functions below; their original success statements/assertions are retained by the resource-delta proof. The right column explains native dependence.

| Test file / original function | Native dependency / scope |
|---|---|
| test_planning_windows_hardening.py::test_closed_uncapped_tree_refuses_before_directory_or_native_create | Direct Windows ownership/launch/retirement or native memory enforcement; original native-body assertions retained. |
| test_planning_windows_hardening.py::test_null_job_refuses_before_membership_or_resume | Direct Windows ownership/launch/retirement or native memory enforcement; original native-body assertions retained. |
| test_planning_windows_hardening.py::test_empty_environment_key_refuses_before_native_create | Direct Windows ownership/launch/retirement or native memory enforcement; original native-body assertions retained. |
| test_planning_windows_hardening.py::test_closed_job_retry_attempts_all_retained_children_and_records_error | Direct Windows ownership/launch/retirement or native memory enforcement; original native-body assertions retained. |
| test_planning_windows_hardening.py::test_repeated_job_close_failures_preserve_first_record | Direct Windows ownership/launch/retirement or native memory enforcement; original native-body assertions retained. |
| test_planning_windows_hardening.py::test_sidecar_write_failure_still_raises_the_handle_error | Direct Windows ownership/launch/retirement or native memory enforcement; original native-body assertions retained. |
| test_planning_windows_launch.py::test_launch_uses_explicit_win32_transaction_not_popen | Direct Windows ownership/launch/retirement or native memory enforcement; original native-body assertions retained. |
| test_planning_windows_launch.py::test_owned_job_membership_precedes_primary_thread_resume | Direct Windows ownership/launch/retirement or native memory enforcement; original native-body assertions retained. |
| test_planning_windows_launch.py::test_create_failure_cannot_publish_or_leave_child | Direct Windows ownership/launch/retirement or native memory enforcement; original native-body assertions retained. |
| test_planning_windows_launch.py::test_resume_failure_or_unexpected_suspend_count_retires | Direct Windows ownership/launch/retirement or native memory enforcement; original native-body assertions retained. |
| test_planning_windows_launch.py::test_failed_job_handle_close_is_not_silently_accepted | Direct Windows ownership/launch/retirement or native memory enforcement; original native-body assertions retained. |
| test_planning_windows_launch.py::test_native_creation_faults_fail_closed | Direct Windows ownership/launch/retirement or native memory enforcement; original native-body assertions retained. |
| test_planning_windows_launch.py::test_unknown_role_or_directory_refuses_before_creation | Direct Windows ownership/launch/retirement or native memory enforcement; original native-body assertions retained. |
| test_planning_windows_launch.py::test_failed_child_handle_cleanup_retains_handle_and_error_evidence | Direct Windows ownership/launch/retirement or native memory enforcement; original native-body assertions retained. |
| test_planning_windows_launch.py::test_invalid_environment_cannot_reach_native_creation | Direct Windows ownership/launch/retirement or native memory enforcement; original native-body assertions retained. |
| test_planning_windows_launch.py::test_incomplete_native_creation_record_refuses_and_retires | Direct Windows ownership/launch/retirement or native memory enforcement; original native-body assertions retained. |
| test_planning_windows_launch.py::test_never_created_child_poll_wait_and_close_are_safe | Direct Windows ownership/launch/retirement or native memory enforcement; original native-body assertions retained. |
| test_planning_memory_process.py::test_real_job_rejects_over_cap_virtual_commit | Direct Windows ownership/launch/retirement or native memory enforcement; original native-body assertions retained. |
| test_planning_memory_process.py::test_tiny_cap_refuses_real_planner_and_cleans | Direct Windows ownership/launch/retirement or native memory enforcement; original native-body assertions retained. |
| test_planning_memory_process.py::test_two_worker_geometry_under_explicit_tree_cap | Actual owned session/workers or host+tree memory/cancellation integration; Windows numerical/profile admission remains additionally required, with all original positive assertions retained. |
| test_planning_host_memory_process.py::test_real_host_and_tree_budget_preserves_small_fixture | Actual owned session/workers or host+tree memory/cancellation integration; Windows numerical/profile admission remains additionally required, with all original positive assertions retained. |
| test_planning_session_loss_process.py::test_session_terminal_loss_retires_without_retry | Actual owned session/workers or host+tree memory/cancellation integration; Windows numerical/profile admission remains additionally required, with all original positive assertions retained. |
| test_planning_session_phase_process.py::test_one_stage_finishes_with_empty_second_worker | Actual owned session/workers or host+tree memory/cancellation integration; Windows numerical/profile admission remains additionally required, with all original positive assertions retained. |
| test_planning_session_process.py::test_public_session_reuses_workers_with_same_samples_and_plan_apply | Actual owned session/workers or host+tree memory/cancellation integration; Windows numerical/profile admission remains additionally required, with all original positive assertions retained. |
| test_planning_session_cancellation_process.py::test_session_callback_after_worker_admission_retires_tree | Actual owned session/workers or host+tree memory/cancellation integration; Windows numerical/profile admission remains additionally required, with all original positive assertions retained. |
| test_planning_cancellation_process.py::test_public_callback_after_worker_admission_retires_tree | Actual owned session/workers or host+tree memory/cancellation integration; Windows numerical/profile admission remains additionally required, with all original positive assertions retained. |
| test_planning_cancellation_process.py::test_false_public_callback_retains_policy_plan_apply_and_source | Actual owned session/workers or host+tree memory/cancellation integration; Windows numerical/profile admission remains additionally required, with all original positive assertions retained. |

## Explicitly untouched portable case

test_planning_session_phase_process.py::test_public_empty_plan_creates_no_owned_session is not wrapped. Its stated property is successful public no-work planning without any OwnedSession or output directory. Any production-profile admission failure there belongs to a separate contract decision; this patch does not declare the body native merely because of its filename. All helper functions in the nine modules are also unchanged.

## Remaining evidence and acceptance

Focused checks, failures and process accounting are recorded in .pp1-evidence/r10-platform-contract/checkpoint.md and summary.json. Windows results and POSIX refusal results must be distinguished via JUnit execution_scope. Body identity proves preservation, not successful native execution. Parent owns independent review, positive-profile CI, POSIX CI, final acceptance and Git integration.


## Owner-approved withdrawn 512MiB resource-safety delta

Only test_planning_memory_process.py::test_two_worker_geometry_under_explicit_tree_cap and test_planning_host_memory_process.py::test_real_host_and_tree_budget_preserves_small_fixture are adapted. Their reuse=True cases still require the entire original parallel-success, numerical/CI, membership, memory and cleanup assertions. Their reuse=False/512MiB configuration was explicitly left unsupported in the R9 closeout. A safe fit still runs every original success assertion; one safe observation does not qualify that configuration.

A terminal is accepted only with exact PlannerCleanupError('session planner cleanup failed') directly caused by the measured owned-tree memory-cap QualificationDecline, exact536870912-byte admission/retirement receipts and0x2200flags, actualpeak>cap, checksummed session/request-bound P/W result explicitly reporting MemoryError, unchanged source, onecandidate/no serial retry, terminal/unverified CI state, zero/verified resource cleanup for planner and workers, and nohandle-cleanup-error artifacts. Unknown source/callback/cleanup context or mismatched/absent evidence fails. No weaker generic allocation-denial branch is implemented. Test-only sys.setprofile observation leaves production bindings unchanged and restores the previous profiler in finally; it is disabled for reuse=True.

Such a result records execution_scope=unsupported512-terminal, memory_bound_accepted=False and parallel_qualified=False. Resource process cleanup verified at zero is distinguished from the failed numeric memory-bound verdict; the original failed-profile receipts are retained. The outer supervisor must separately establish zero remaining owned processes before acceptance. Portable validator fixtures exercise source/callback/cleanup/context corruption and invalid flags/limits/peak/active/verified receipts; they are labeled resource-evidence-validation-unit and make no native execution claim.

The earlier body-identity-proof.json remains historical proof of the first wrapper-only patch. resource-body-preservation-proof.json supersedes its current-source claim: seven files/other25 native Windows bodies stay exact; the two approved functions retain every original success statement/assertion, parametrization and512MiB limit with only the narrow success-or-strict-terminal control-flow addition. Current focused results and unmet gates remain in the task checkpoint.

## Declared numerical worker profile and independent resource ABI

The positive numerical P/W integration profile is Windows CPython 3.14.2 / NumPy 2.4.6 / SciPy 1.17.1, with the existing immutable production source/native image pins. It is separate from runtime admission and scientific qualification. The resource helper's ABI3.11 native ownership, explicit handle inheritance, suspended creation, Job membership, retirement and raw memory-enforcement tests remain independently required on Windows. This matrix does not change production admission, generated-module origin requirements, native/source pins, numerical tolerances or scientific gates.

Evidence `r10-native-runtime313-diagnosis01` records three real failures on a different Windows 3.13 runtime: the public positive declined to serial, the reusable public cancellation did not reach admission, and a direct numerical OwnedPlanner stage refused with `module has no qualified origin: _cython_3_1_6`. These are initial admission failures, not observations of the intended numerical or cancellation behavior. They do not justify relaxing generated-module origin checks. Conversely, an immutable context-image refusal does not establish that every distinct legacy runtime path is unsupported.

For only the functions listed below, `windows_owned_test` is the outermost execution adapter and `numpy_profile_test` is inside it. Parametrization remains outside both. POSIX therefore executes the real exact OwnedTree refusal before any profile classification. On Windows, an accepted immutable context image executes the unchanged original function. Other images execute the real exact typed context refusal and verify unchanged settings through the existing owner-load getter anchors. JUnit records `initial-profile-refusal`, `original_test_body_executed=False`, `numerical_worker_execution=False`, `postcapture_qualification=False`, and `legacy_runtime_admission=not-evaluated`. This result covers only the declared-profile boundary; it does not claim the original worker, phase, cancellation or legacy admission path ran.

The mandatory pinned positive Windows job sets `ANYGEOMETRY_REQUIRE_QUALIFIED_PLANNING_TESTS=1`. Every platform/profile/runtime/dispatcher helper fails if it would otherwise classify an initial refusal. A portable flag-guard unit substitutes only the platform helper's OS namespace and a fail-fast ownership boundary; it does not fake WinAPI or claim native execution. Thus all original numerical assertions, including the 22 core counter cases and the full suite's additional positive gates, must execute on that lane. Generic public serial mathematics remains separately required. No file-wide decorators, skips, xfails or catch-all acceptance are introduced. Contract review and final CI acceptance remain with the parent owner; Mistral review precedes adoption/main delivery.

The explicit addition covers **27 cases in 17 functions**: 25 numerical worker-process cases plus the two identified H-positive cases. Both reuse variants require admitted numerical P/W before their resource or callback assertions are meaningful; comparison loops retain both routes in their original body.

| File | Original function | Cases |
|---|---|---:|
| test_planning_cancellation_process.py | test_public_callback_after_worker_admission_retires_tree | 4 |
| test_planning_cancellation_process.py | test_false_public_callback_retains_policy_plan_apply_and_source | 1 |
| test_planning_host_memory_process.py | test_real_host_and_tree_budget_preserves_small_fixture | 2 |
| test_planning_memory_process.py | test_two_worker_geometry_under_explicit_tree_cap | 2 |
| test_planning_payload_process.py | test_real_compact_preparation_with_memory_bounded_startup | 1 |
| test_planning_real_process.py | test_real_two_workers_transport_sampling_ci_cleanup | 1 |
| test_planning_real_process.py | test_real_nested_cancellation_never_replays_serial | 1 |
| test_planning_session_cancellation_process.py | test_session_callback_after_worker_admission_retires_tree | 4 |
| test_planning_session_loss_process.py | test_session_terminal_loss_retires_without_retry | 2 |
| test_planning_session_partial_process.py | test_disjoint_pairs_replay_sample_dependent_support_state | 1 |
| test_planning_session_phase_process.py | test_one_stage_finishes_with_empty_second_worker | 2 |
| test_planning_session_process.py | test_public_session_reuses_workers_with_same_samples_and_plan_apply | 1 |
| test_planning_session_signal_process.py | test_session_false_signal_retains_policy_plan_apply_and_source | 1 |
| test_planning_stage_process.py | test_owned_stage_workers_decode_once_and_match_serial | 1 |
| test_planning_stage_process.py | test_real_requested_payload_limit_exceeds_default | 1 |
| test_planning_host.py | test_public_real_pair_and_face_stages_and_apply_ci | 1 |
| test_planning_stage_package.py | test_admitted_package_capability_and_manifest_skew_are_terminal | 1 |

The raw over-cap virtual-commit test, tiny-cap pre-spawn denial, strict resource-evidence validator, corrupt-checksum transport refusal, started-invalid-model fallback, public empty plan, and direct native handle/launch/ownership tests are outside this numerical-profile addition. Their Windows bodies remain independent of the numerical image classifier.

Before-source identities, per-function counts and exact original function source are saved in `.pp1-evidence/r10-numerical-profile-matrix/assignment-before.json` and `assignment-map.json`. The corresponding body proof compares exact function text and AST, and whole-module AST after removing the newly added imports/adapters and the one explicitly added portable flag-guard unit. This proof establishes preservation only; representative process checks and the later full CI establish their own execution evidence.
