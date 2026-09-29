---
name: deprecate-and-migrate
description: Change or remove public API, CLI flags or stored formats without breaking users - deprecation warnings, a compatibility period, migrations. Use when a change renames, removes or changes the meaning of anything users or their saved data depend on.
---

# Deprecate and migrate

Users and their saved data outlive any version. Break nothing silently.

**Public API and CLI**
1. Keep the old name working for at least one minor release: it calls the new one and warns with
   `warnings.warn("<old> is deprecated, use <new>; it will be removed in <version>", DeprecationWarning, stacklevel=2)`.
   CLI: the same message on stderr.
2. Docs and examples show only the new form. `CHANGELOG.md`: **Deprecated** now, **Removed** when it goes.
3. Tests: the old form still works and warns; the new form works.

**Saved profiles and reference sets** (`design/current.md` §4.6, §4.7)
1. Profiles (`.hone/taste/profiles/<name>.json`) and reference sets outlive versions. New optional
   fields are fine; anything else is a change record with a "Migration and compatibility" section.
2. An old file still loads, or is refused with a `HoneTasteError` that says how to migrate. Never guess.
3. A changed reference set changes every score scaled by it: say so in `CHANGELOG.md`, and regenerate it
   with `scripts/make_reference_sets.py`.
4. Keep a small old-format file in `tests/fixtures/` and test loading it and the refusal.
