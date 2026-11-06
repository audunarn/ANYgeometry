# R5 planning contract-test migration

Authority: user-approved test migration; source/numerical/native guards, pins and workflows are unchanged. Original baseline .pp1-evidence/r10-historical-contract01 remains retained: 69 cases, 39 failed, 30 passed. R9 impact inventory classified 61 cases superseded, leaving eight packaging/build cases; a historical pass does not prove that its intended property ran beyond an initial stale-profile refusal.

All five originals are byte-for-byte .py.txt siblings; archive-index.json binds their SHA256 hashes. Rows below cover every original test function and all its parameter variants. Replacements consolidate properties rather than preserve obsolete parametrization counts. New exact counts/results are recorded in .pp1-evidence/r10-contract-migration/checkpoint.md after verification.

Current claim: deep bind anygeometry-owned definitions, retain imported external module/symbol/settings boundaries, preserve actual value-wire and native ownership gates. No whole-interpreter purity, private codec-cache/search/error-handler ownership, native audit-list absence, or Plane-selected-body exemption is claimed. Diagnostic inspector provenance/build gates retain explicit scope.

Execution classification: runtime-image-bound CallableManifest captures have an explicit Windows-only adapter in current production, after availability of the NumPy dispatcher namespace. _planning_contract_helpers.capture_with_runtime_boundary asserts the actual precise typed refusal for the known missing NumPy dispatcher or other platforms and returns no manifest; no later mutation/positive assertion runs there. Windows admitted captures retain positive and mutation checks. The one loaded-package/settings test independently compares immutable NumPy source/native hashes; unsupported images execute exact typed refusal; admitted images run real capture/verification/settings mutation. Optional private-inspector tests retain actual native checks only when the binary is installed, otherwise real InspectionUnavailable. These paths do not claim native, parallel or postcapture execution on initial refusal. No skip/xfail, tolerance change or pin update is introduced. The migration helper accepts only the explicitly identified unsupported-platform/runtime refusals; it does not suppress arbitrary capture errors. Existing production validators may still convert post-capture validation failures to terminal PlannerSourceChanged while preserving the original exception as __cause__; that fail-closed behavior is unchanged.

## test_planning_branch_profiles.py

| Archived test/property (all variants) | Current replacement or honest boundary |
|---|---|
| test_plane_face_selectors_are_proved_without_constructor_call | Obsolete diagnostic Plane-selector constructor exclusion; no production exclusion claimed. Actual selector results/model preservation: guarded_capture::test_actual_selector_results_preserve_the_model. Owned definition drift: branch_profiles::test_all_owned_methods_remain_bound_even_when_unselected. |
| test_actual_domain_support_not_nominal_face_label_controls_pair | branch_profiles::test_actual_domain_support_and_freeze_preserve_model uses actual three-Plane fixture, not the obsolete plane-cylinder label. No mixed-shape classification claim. |
| test_declarative_plane_face_guard_preserves_model | branch_profiles::test_actual_domain_support_and_freeze_preserve_model; guarded_capture::test_actual_selector_results_preserve_the_model retain actual model preservation without branch authority. |
| test_default_norm_profile_covers_exact_float64_any_dimension | branch_profiles::test_stage_codec_retains_exact_float64_array_shape_and_bits retains four shapes at the actual stage-value boundary; no NumPy norm-internals exclusion claimed. |
| test_norm_other_variants_refuse | branch_profiles::test_stage_codec_refuses_arrays_outside_its_schema checks actual dtype/nonfinite/layout refusal. Norm axis/ord/keepdims restrictions are no longer claimed by production; NumPy imported interfaces are pinned, not restricted by Plane profiles. |
| test_json_profiles_bind_exact_effective_kwargs_and_closed_tree | json_operation::test_closed_stage_value_survives_actual_json_transport and test_current_wire_refuses_nonfinite_or_unknown_values retain actual transport and closed value semantics. No JSON-internals purity/profile waiver. |
| test_profile_detects_json_method_rebinding | runtime_audit::test_owned_caller_imported_numpy_binding_drift_is_terminal tests the current referenced imported interface binding. Arbitrary JSONEncoder internals are explicitly outside the claimed boundary. |
| test_profile_rejects_array_subclass_without_hooks | branch_profiles::test_array_subclass_refuses_without_dispatching_hooks retains actual stage-codec refusal without hook dispatch. |
| test_initial_same_module_function_substitution_refuses | branch_profiles::test_obsolete_diagnostic_source_pin_is_still_rejected retains exact stale-source rejection; test_all_owned_methods_remain_bound_even_when_unselected and test_owned_code_identity_change_is_terminal_even_for_equal_behavior cover current owned definition drift. Initial private profile authority is retired. |
| test_non_native_endian_nan_is_not_read_as_native_finite_bytes | branch_profiles::test_non_native_endian_nan_is_not_admitted_as_finite_native_bytes retains actual stage schema refusal. |
| test_initial_json_native_replacement_refuses | No equivalent production assertion over unreferenced JSON encoder internals. Imported referenced symbol/module binding is covered in runtime_audit; current value schema remains enforced. |
| test_initial_surface_descriptor_replacement_does_not_execute | Private Plane-profile initial descriptor attestation is not production authority. Existing loaded_manifest::test_unknown_class_property_is_not_executed and finite_dispatch metadata-hook tests retain non-dispatch gates; current owned definition postcapture drift remains covered. |
| test_initial_predicate_alias_or_builtin_shadow_refuses | Diagnostic initial alias whitelist retired; source-pin refusal remains explicit. Deep binding of current owned methods/code is tested without selected-body exclusions. Arbitrary whole-interpreter builtin purity is not claimed. |
| test_face_namespace_custom_mapping_never_executes | Private declarative Plane namespace classifier retired; actual model/constructor refusal contracts remain in test_planning_host_constructor.py and loaded_manifest::test_unknown_state_refuses (unchanged). No custom-namespace profile admission claim. |

## test_planning_codec_registry.py

| Archived test/property (all variants) | Current replacement or honest boundary |
|---|---|
| test_inspector_only_is_nonlimited_and_keeps_numerical_core_off | RETAINED packaging assertions; Windows emits only nonlimited inspector, other platforms assert optional-off and exact required-helper rejection. No skipped nodes. |
| test_combined_helper_build_preserves_old_abi_and_uses_nonlimited_wheel | RETAINED native ABI separation; Windows stable resource helper plus nonlimited inspector, other platforms assert optional-off/required native refusal. |
| test_default_off_keeps_extensions_absent | RETAINED unchanged default-off extension assertion. |
| test_passive_inspector_api_returns_immutable_container_copies | Private codec search/error/cache container observation and pinned CPython DLL contents are no longer production claims. Current codec_registry::test_current_value_codec_contract_does_not_require_private_inspector verifies actual owned value codec request/schema/transport. |
| test_actual_mixed_wheel_tag_is_not_abi3 | RETAINED real bdist_wheel tag check with current interpreter tag instead of hard-coded cp314; pure build tag asserted outside Windows. |
| test_observer_does_not_search_or_fill_missing_selected_cache | No production claim to passively inspect CPython codec caches or forbid provider cache warming; current codec contract tested without importing private inspector. |
| test_cached_entries_and_search_mutation_are_reported_by_identity | No codec-search-list/cache ownership claim. Referenced imported symbol/code drift tested in runtime_audit; request/schema mutation in test_current_value_codec_request_binding_cannot_be_forged. |
| test_cache_warming_is_drift_without_policy_exception | Codec warming is outside package-definition ownership, no equivalent purity claim. runtime_audit::test_unrelated_interpreter_module_is_not_a_package_owned_definition demonstrates explicit scope, not codec-state acceptance proof. |
| test_error_handler_identity_change_and_strong_lifetime | Global codec error-handler lifetime/private-registry ownership is not claimed. No equivalent global-lifetime assertion; current request binding and external interface pins remain. |
| test_altered_exports_refused_before_call | codec_registry::test_altered_diagnostic_exports_refuse_before_call retains optional diagnostic export refusal. Missing helper exercises actual InspectionUnavailable; does not claim an altered native export ran. |
| test_missing_helper_does_not_rebaseline | RETAINED exact missing-anchor refusal; no runtime inspector rebaseline. |
| test_replaced_module_and_loader_decline | codec_registry::test_replaced_diagnostic_module_and_loader_decline retains optional native provenance refusal, with truthful unavailable-helper branch. |
| test_profile_layout_and_compiler_identities_are_distinct | Private CPython codec offsets/compiler/layout interpretation is outside production contract. Native stable ABI/build isolation remains in retained packaging tests; no substitute private-layout proof. |
| test_native_rejects_foreign_capsule_and_guard_has_no_override | codec_registry::test_native_rejects_foreign_capsule_and_has_no_override retains actual native capsule/signature checks when installed; otherwise exact helper-unavailable refusal, not native execution. |
| test_initial_capture_refuses_altered_export_without_invoking | codec_registry::test_initial_diagnostic_capture_refuses_altered_export_without_invoking retains initial native-export provenance rejection when installed; actual unavailable refusal otherwise. |
| test_hostile_unselected_key_never_dispatches_equality_and_present_none_is_distinct | No direct access to interpreter-private codec registry or exact present-None representation is claimed. Closed input refusal without foreign hooks is tested at actual stage-codec boundary. |
| test_build_profile_rejects_mismatched_inputs_before_compilation | RETAINED four diagnostic build-input refusal variants; exact supported runtime tests altered version/headers/DLL/flags, unsupported runtimes assert initial exact-profile refusal without a compilation claim. |
| test_matches_rechecks_after_allocating_wrapper_validation | Private interpreter snapshot absence/matches freshness is not claimed. Current owned manifest drift tests and value-codec request binding remain; no equivalent codec-snapshot proof. |

## test_planning_json_operation.py

| Archived test/property (all variants) | Current replacement or honest boundary |
|---|---|
| test_closed_json_operation_capture_and_actual_call | json_operation::test_closed_stage_value_survives_actual_json_transport executes both ASCII modes and verifies shared-reference reconstruction; no guarded JSON-internals authority. |
| test_guarded_json_rechecks_tree_before_invocation | test_unknown_object_added_before_encoding_refuses_without_hooks exercises actual closed input schema after mutation; does not claim obsolete profile pre-invocation attestation. |
| test_guarded_json_rechecks_guard_definition | test_unknown_codec_version_cannot_bypass_the_current_transport and codec_registry::test_current_value_codec_request_binding_cannot_be_forged retain current version/schema refusal. Owned code identity drift remains in branch_profiles. No diagnostic BranchProfiles guard claim. |
| test_json_operation_declines_integer_conversion_context_dependence | Old 2048-bit integer profile restriction is not a production transport contract; test_large_integer_uses_current_wire_contract_not_obsolete_json_profile exercises exact current integer roundtrip. Interpreter integer-conversion purity is not claimed. |
| test_initial_encoder_assignment_hooks_refuse_without_execution | Initial JSONEncoder assignment-hook attestation is outside production external-library trust boundary. Closed unknown values/hooks refuse at actual owned value-codec boundary; imported referenced symbol drift is separately tested. |
| test_json_operation_keeps_four_megabyte_bound | test_actual_stage_decode_retains_four_megabyte_payload_limit tests the unchanged 4,000,000-byte production decode boundary. |
| test_ascii_surrogate_pairs_are_counted_as_twelve_bytes | test_ascii_surrogate_expansion_is_bounded_at_actual_stage_decode executes actual JSON expansion for both key and value, observes actual wire >4MB and checks unchanged decode refusal. No profile size estimator claim. |
| test_non_ascii_profile_uses_utf8_bound_for_supplementary_characters | test_non_ascii_stage_transport_preserves_supplementary_characters exercises actual non-ASCII stage JSON/decode; fixed obsolete profile byte count no longer claimed. |

## test_planning_runtime_audit.py

| Archived test/property (all variants) | Current replacement or honest boundary |
|---|---|
| test_actual_audit_hooks_refuse_without_invocation | runtime_audit::test_benign_python_audit_observer_does_not_imply_a_purity_claim executes before/after Python observers in disposable children. No native hook-list inspection, callback noninvocation, global audit absence, or allocation-window absence proof claimed. Referenced interface mutation is separately tested. |
| test_absence_and_audit_layout_are_bound | test_imported_external_module_is_bound_to_its_actual_source_image verifies current module-file provenance. Private CPython audit offsets/absence schema are explicitly no longer claimed. |

## test_planning_guarded_capture.py

| Archived test/property (all variants) | Current replacement or honest boundary |
|---|---|
| test_guarded_selector_capture_and_actual_call | guarded_capture::test_actual_selector_results_preserve_the_model executes all three current package selectors, preserves result identity/model state. No <100-node Plane exclusion claimed. |
| test_guarded_selector_checks_argument_domain_again_before_call | test_opaque_callable_state_is_refused_without_execution retains no foreign callable invocation; owned selected/unselected definition mutation remains in branch_profiles. The diagnostic selector argument guard is no longer a production authority; existing constructor/state tests remain untouched. |
| test_uncompleted_profile_does_not_exclude_dependency_body | branch_profiles::test_all_owned_methods_remain_bound_even_when_unselected proves no selected-body exclusion; test_obsolete_diagnostic_source_pin_is_still_rejected preserves the old exact pin guard. |
