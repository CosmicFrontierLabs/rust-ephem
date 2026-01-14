# Changelog

## [0.3.1](https://github.com/CosmicFrontierLabs/rust-ephem/compare/v0.3.0...v0.3.1) (2026-01-14)


### Bug Fixes

* add more of this goodness ([6e761e4](https://github.com/CosmicFrontierLabs/rust-ephem/commit/6e761e40bd123055f4151fdc2f1f3f85ad344ab4))


### Performance Improvements

* add cosine opt to other constraints ([#110](https://github.com/CosmicFrontierLabs/rust-ephem/issues/110)) ([6c40a45](https://github.com/CosmicFrontierLabs/rust-ephem/commit/6c40a457f3e1d2f0925a9631a1a898a485b1daac))
* increase speed of airmass constraint calculation ([#111](https://github.com/CosmicFrontierLabs/rust-ephem/issues/111)) ([382edc1](https://github.com/CosmicFrontierLabs/rust-ephem/commit/382edc18307d5ec733a0afb273c489830b168da6))
* O(1) index lookup in in_constraint() ([d891d0b](https://github.com/CosmicFrontierLabs/rust-ephem/commit/d891d0b79f04ed180d62ebf106eeb02254e4e637))
* O(1) index lookup in in_constraint() instead of O(n) linear search ([820fc20](https://github.com/CosmicFrontierLabs/rust-ephem/commit/820fc20a56517a6134d901bc52b90880b0c0d2a2))
* remove redundant array clones in constraint macros ([#107](https://github.com/CosmicFrontierLabs/rust-ephem/issues/107)) ([bc18402](https://github.com/CosmicFrontierLabs/rust-ephem/commit/bc184020ae6ce1a0f648c315d3dc9b1ff26ddf67))
* use cosine threshold instead of acos in batch constraint evaluation ([365d206](https://github.com/CosmicFrontierLabs/rust-ephem/commit/365d20684e385724d1e5fb830af6fa5defc94668))
* use cosine threshold instead of acos in batch constraint evaluation ([b158a2c](https://github.com/CosmicFrontierLabs/rust-ephem/commit/b158a2c5eefac94463ab46ef1387f96ddd25477b))

## [0.3.0](https://github.com/CosmicFrontierLabs/rust-ephem/compare/v0.2.1...v0.3.0) (2026-01-02)


### Features

* add sun/moon/earth ra/dec values in degrees and radians ([#105](https://github.com/CosmicFrontierLabs/rust-ephem/issues/105)) ([81bf0ae](https://github.com/CosmicFrontierLabs/rust-ephem/commit/81bf0aec788fc59929f6a018a311a0a2a6dd1075))


### Bug Fixes

* add dependabot for actions ([#97](https://github.com/CosmicFrontierLabs/rust-ephem/issues/97)) ([fd394c6](https://github.com/CosmicFrontierLabs/rust-ephem/commit/fd394c6f5be16187915ddf83ac1152a0c9c36f0a))
* fix image and changelog ([#104](https://github.com/CosmicFrontierLabs/rust-ephem/issues/104)) ([9ccff1b](https://github.com/CosmicFrontierLabs/rust-ephem/commit/9ccff1b050f4b7acab24ad674c32bcdcc2052ca1))

## [0.2.1](https://github.com/CosmicFrontierLabs/rust-ephem/compare/v0.2.0...v0.2.1) (2025-12-18)


### Bug Fixes

* don't include name of repo in tag ([#95](https://github.com/CosmicFrontierLabs/rust-ephem/issues/95)) ([ea24511](https://github.com/CosmicFrontierLabs/rust-ephem/commit/ea24511edac5b49f7969f88418e6208f57d33870))

[Changes][v0.2.1]


<a id="v0.2.0"></a>
## [v0.2.0: Add release-please](https://github.com/CosmicFrontierLabs/rust-ephem/releases/tag/v0.2.0) - 2025-12-18

### What's Changed
* Fix docs errors by [@jak574](https://github.com/jak574) in [#90](https://github.com/CosmicFrontierLabs/rust-ephem/pull/90)
* feat: add `release-please` versioning by [@jak574](https://github.com/jak574) in [#92](https://github.com/CosmicFrontierLabs/rust-ephem/pull/92)
* fix: use v in tag name by [@jak574](https://github.com/jak574) in [#94](https://github.com/CosmicFrontierLabs/rust-ephem/pull/94)
* chore(main): release rust_ephem 0.2.0 by [@jak574](https://github.com/jak574) in [#93](https://github.com/CosmicFrontierLabs/rust-ephem/pull/93)


**Full Changelog**: https://github.com/CosmicFrontierLabs/rust-ephem/compare/v0.1.15...v0.2.0

[Changes][v0.2.0]


<a id="v0.1.15"></a>
## [v0.1.15: additional constraints and solar system body tracking](https://github.com/CosmicFrontierLabs/rust-ephem/releases/tag/v0.1.15) - 2025-12-11

### What's Changed
* Add more constraints by [@jak574](https://github.com/jak574) in [#87](https://github.com/CosmicFrontierLabs/rust-ephem/pull/87)
* Make `ConstraintResult` a Pydantic model by [@jak574](https://github.com/jak574) in [#88](https://github.com/CosmicFrontierLabs/rust-ephem/pull/88)
* Solar system body visibility by [@jak574](https://github.com/jak574) in [#89](https://github.com/CosmicFrontierLabs/rust-ephem/pull/89)


**Full Changelog**: https://github.com/CosmicFrontierLabs/rust-ephem/compare/v0.1.14...v0.1.15

[Changes][v0.1.15]


<a id="v0.1.14"></a>
## [v0.1.14: Refactors and re-orgs](https://github.com/CosmicFrontierLabs/rust-ephem/releases/tag/v0.1.14) - 2025-12-07

### What's Changed
* Docs fixes by [@jak574](https://github.com/jak574) in [#69](https://github.com/CosmicFrontierLabs/rust-ephem/pull/69)
* Build(deps): Bump urllib3 from 2.5.0 to 2.6.0 by [@dependabot](https://github.com/dependabot)[bot] in [#71](https://github.com/CosmicFrontierLabs/rust-ephem/pull/71)
* Revamp of TLE handling by [@jak574](https://github.com/jak574) in [#73](https://github.com/CosmicFrontierLabs/rust-ephem/pull/73)
* Reduce boilerplate by [@jak574](https://github.com/jak574) in [#76](https://github.com/CosmicFrontierLabs/rust-ephem/pull/76)
* `hifitime` refactor by [@jak574](https://github.com/jak574) in [#70](https://github.com/CosmicFrontierLabs/rust-ephem/pull/70)
* Add API documentation for `fetch_tle()` by [@jak574](https://github.com/jak574) in [#79](https://github.com/CosmicFrontierLabs/rust-ephem/pull/79)
* Fix typing issue on `fetch_tle` `enforce_source` argument by [@jak574](https://github.com/jak574) in [#81](https://github.com/CosmicFrontierLabs/rust-ephem/pull/81)
* Reduce length and scope of README.md by [@jak574](https://github.com/jak574) in [#82](https://github.com/CosmicFrontierLabs/rust-ephem/pull/82)
* Add local multi platform build scripts by [@jak574](https://github.com/jak574) in [#75](https://github.com/CosmicFrontierLabs/rust-ephem/pull/75)
* Change tagline, get rid of broken `pre-commit` hook by [@jak574](https://github.com/jak574) in [#83](https://github.com/CosmicFrontierLabs/rust-ephem/pull/83)
* Remove deprecated `evaluate_batch` method. Allow in_constraint to take time arrays. by [@jak574](https://github.com/jak574) in [#85](https://github.com/CosmicFrontierLabs/rust-ephem/pull/85)

## New Contributors
* [@dependabot](https://github.com/dependabot)[bot] made their first contribution in [#71](https://github.com/CosmicFrontierLabs/rust-ephem/pull/71)

**Full Changelog**: https://github.com/CosmicFrontierLabs/rust-ephem/compare/v0.1.13...v0.1.14

[Changes][v0.1.14]


<a id="v0.1.13"></a>
## [v0.1.13: Documentation updates, plus type fixes](https://github.com/CosmicFrontierLabs/rust-ephem/releases/tag/v0.1.13) - 2025-12-04

### What's Changed
* Read the Docs config plus Docs update by [@jak574](https://github.com/jak574) in [#63](https://github.com/CosmicFrontierLabs/rust-ephem/pull/63)
* Add missing requirements.txt for docs build by [@jak574](https://github.com/jak574) in [#64](https://github.com/CosmicFrontierLabs/rust-ephem/pull/64)
* More docs updates by [@jak574](https://github.com/jak574) in [#65](https://github.com/CosmicFrontierLabs/rust-ephem/pull/65)
* More docs updates by [@jak574](https://github.com/jak574) in [#66](https://github.com/CosmicFrontierLabs/rust-ephem/pull/66)
* fix(types): fix mismatched types in _rust_ephem.pyi by [@jak574](https://github.com/jak574) in [#68](https://github.com/CosmicFrontierLabs/rust-ephem/pull/68)


**Full Changelog**: https://github.com/CosmicFrontierLabs/rust-ephem/compare/v0.1.12...v0.1.13

[Changes][v0.1.13]


<a id="v0.1.12"></a>
## [v0.1.12: Bug fix on type hint](https://github.com/CosmicFrontierLabs/rust-ephem/releases/tag/v0.1.12) - 2025-12-03

### What's Changed
* Fix `step_size` hinting by [@jak574](https://github.com/jak574) in [#62](https://github.com/CosmicFrontierLabs/rust-ephem/pull/62)


**Full Changelog**: https://github.com/CosmicFrontierLabs/rust-ephem/compare/v0.1.11...v0.1.12

[Changes][v0.1.12]


<a id="v0.1.11"></a>
## [v0.1.11: Fix type hinting issues](https://github.com/CosmicFrontierLabs/rust-ephem/releases/tag/v0.1.11) - 2025-12-03

### What's Changed
* Fix type hinting issues by [@jak574](https://github.com/jak574) in [#61](https://github.com/CosmicFrontierLabs/rust-ephem/pull/61)


**Full Changelog**: https://github.com/CosmicFrontierLabs/rust-ephem/compare/v0.1.10...v0.1.11

[Changes][v0.1.11]


<a id="v0.1.10"></a>
## [v0.1.10: Update to Typing Support](https://github.com/CosmicFrontierLabs/rust-ephem/releases/tag/v0.1.10) - 2025-12-01

### What's Changed
* Add generic `Ephemeris` type for type checking by [@jak574](https://github.com/jak574) in [#59](https://github.com/CosmicFrontierLabs/rust-ephem/pull/59)


**Full Changelog**: https://github.com/CosmicFrontierLabs/rust-ephem/compare/v0.1.9...v0.1.10

[Changes][v0.1.10]


<a id="v0.1.9"></a>
## [v0.1.9: More optimizations and API name change](https://github.com/CosmicFrontierLabs/rust-ephem/releases/tag/v0.1.9) - 2025-11-26

### What's Changed
* Improve vectorization and update API naming scheme by [@jak574](https://github.com/jak574) in [#57](https://github.com/CosmicFrontierLabs/rust-ephem/pull/57)


**Full Changelog**: https://github.com/CosmicFrontierLabs/rust-ephem/compare/v0.1.8...v0.1.9

[Changes][v0.1.9]


<a id="v0.1.8"></a>
## [v0.1.8: Vectorized constraint calculations](https://github.com/CosmicFrontierLabs/rust-ephem/releases/tag/v0.1.8) - 2025-11-26

### What's Changed
* Potential fix for code scanning alert no. 1: Workflow does not contain permissions by [@jak574](https://github.com/jak574) in [#54](https://github.com/CosmicFrontierLabs/rust-ephem/pull/54)
* Add vectorized constraint evaluation by [@jak574](https://github.com/jak574) in [#56](https://github.com/CosmicFrontierLabs/rust-ephem/pull/56)


**Full Changelog**: https://github.com/CosmicFrontierLabs/rust-ephem/compare/v0.1.7...v0.1.8

[Changes][v0.1.8]


<a id="v0.1.7"></a>
## [v0.1.7: Reflect parameters back in Ephemeris objects](https://github.com/CosmicFrontierLabs/rust-ephem/releases/tag/v0.1.7) - 2025-11-21

### What's Changed
* Reflect Ephemeris Parameters back to user in objects by [@jak574](https://github.com/jak574) in [#53](https://github.com/CosmicFrontierLabs/rust-ephem/pull/53)


**Full Changelog**: https://github.com/CosmicFrontierLabs/rust-ephem/compare/v0.1.6...v0.1.7

[Changes][v0.1.7]


<a id="v0.1.6"></a>
## [v0.1.6: Simplify TLE ingest](https://github.com/CosmicFrontierLabs/rust-ephem/releases/tag/v0.1.6) - 2025-11-21

### What's Changed
* Simplify TLE reading in TLEEphemeris with file, URL, and Celestrak support by [@Copilot](https://github.com/Copilot) in [#49](https://github.com/CosmicFrontierLabs/rust-ephem/pull/49)
* fix: missing entries in pyi files by [@jak574](https://github.com/jak574) in [#51](https://github.com/CosmicFrontierLabs/rust-ephem/pull/51)


**Full Changelog**: https://github.com/CosmicFrontierLabs/rust-ephem/compare/v0.1.5...v0.1.6

[Changes][v0.1.6]


<a id="v0.1.5"></a>
## [v0.1.5: Add latitude / longitude / height values to Ephemeris](https://github.com/CosmicFrontierLabs/rust-ephem/releases/tag/v0.1.5) - 2025-11-20

### What's Changed
* Add latitude / longitude / height to all Ephemeris types by [@jak574](https://github.com/jak574) in [#46](https://github.com/CosmicFrontierLabs/rust-ephem/pull/46)


**Full Changelog**: https://github.com/CosmicFrontierLabs/rust-ephem/compare/v0.1.4...v0.1.5

[Changes][v0.1.5]


<a id="v0.1.4"></a>
## [v0.1.4: Add OEM support](https://github.com/CosmicFrontierLabs/rust-ephem/releases/tag/v0.1.4) - 2025-11-18

### What's Changed
* bugfix: update types on Ephemeris properties by [@jak574](https://github.com/jak574) in [#38](https://github.com/CosmicFrontierLabs/rust-ephem/pull/38)
* feat: add support for CCDSD EOM files by [@jak574](https://github.com/jak574) in [#40](https://github.com/CosmicFrontierLabs/rust-ephem/pull/40)
* Add support for xor for constraints by [@jak574](https://github.com/jak574) in [#44](https://github.com/CosmicFrontierLabs/rust-ephem/pull/44)
* bugfix: add in missing OEMEphemeris definition in __init__.py by [@jak574](https://github.com/jak574) in [#42](https://github.com/CosmicFrontierLabs/rust-ephem/pull/42)


**Full Changelog**: https://github.com/CosmicFrontierLabs/rust-ephem/compare/v0.1.3...v0.1.4

[Changes][v0.1.4]


<a id="v0.1.3"></a>
## [v0.1.3: Bug fix in constraint / visibility calculations](https://github.com/CosmicFrontierLabs/rust-ephem/releases/tag/v0.1.3) - 2025-11-17

### What's Changed
* feat: add index method to ephemeris by [@jak574](https://github.com/jak574) in [#29](https://github.com/CosmicFrontierLabs/rust-ephem/pull/29)
* Binary search for `index()` by [@jak574](https://github.com/jak574) in [#31](https://github.com/CosmicFrontierLabs/rust-ephem/pull/31)
* Don't let earth radius go to infinity by [@jak574](https://github.com/jak574) in [#33](https://github.com/CosmicFrontierLabs/rust-ephem/pull/33)
* bugfix: fix inverted constraints by [@jak574](https://github.com/jak574) in [#35](https://github.com/CosmicFrontierLabs/rust-ephem/pull/35)


**Full Changelog**: https://github.com/CosmicFrontierLabs/rust-ephem/compare/v0.1.2...v0.1.3

[Changes][v0.1.3]


<a id="v0.1.2"></a>
## [v0.1.2: Fix versioning. Add  Earth/Sun/Moon angular size](https://github.com/CosmicFrontierLabs/rust-ephem/releases/tag/v0.1.2) - 2025-11-16

### What's Changed
* Add properties for Earth, Sun, Moon angular radii by [@jak574](https://github.com/jak574) in [#27](https://github.com/CosmicFrontierLabs/rust-ephem/pull/27)
* Fix version extraction from git tags in build workflow by [@Copilot](https://github.com/Copilot) in [#25](https://github.com/CosmicFrontierLabs/rust-ephem/pull/25)

## New Contributors
* [@Copilot](https://github.com/Copilot) made their first contribution in [#25](https://github.com/CosmicFrontierLabs/rust-ephem/pull/25)

**Full Changelog**: https://github.com/CosmicFrontierLabs/rust-ephem/compare/v0.1.1...v0.1.2

[Changes][v0.1.2]


<a id="v0.1.1"></a>
## [v0.1.1: Refactors, optimizations, new methods, github CI](https://github.com/CosmicFrontierLabs/rust-ephem/releases/tag/v0.1.1) - 2025-11-16

### What's Changed
* feat: add pre-commit to repo by [@jak574](https://github.com/jak574) in [#5](https://github.com/CosmicFrontierLabs/rust-ephem/pull/5)
* feat: add github CI actions by [@jak574](https://github.com/jak574) in [#6](https://github.com/CosmicFrontierLabs/rust-ephem/pull/6)
* feat: add typing support to python package by [@jak574](https://github.com/jak574) in [#10](https://github.com/CosmicFrontierLabs/rust-ephem/pull/10)
* fix: fix issues with `ConstraintResult` `in_constraint` and `timestamp` by [@jak574](https://github.com/jak574) in [#8](https://github.com/CosmicFrontierLabs/rust-ephem/pull/8)
* Refactor code and directory structure by [@jak574](https://github.com/jak574) in [#12](https://github.com/CosmicFrontierLabs/rust-ephem/pull/12)
* Optimizations and improvements to calculating whether a target is constraint at a given time. by [@jak574](https://github.com/jak574) in [#14](https://github.com/CosmicFrontierLabs/rust-ephem/pull/14)
* Make `SkyCoord` generation lazy to make initial `Ephemeris` generation quicker. by [@jak574](https://github.com/jak574) in [#16](https://github.com/CosmicFrontierLabs/rust-ephem/pull/16)
* Put in place infrastructure to publish to PyPI by [@jak574](https://github.com/jak574) in [#18](https://github.com/CosmicFrontierLabs/rust-ephem/pull/18)
* More github action work by [@jak574](https://github.com/jak574) in [#19](https://github.com/CosmicFrontierLabs/rust-ephem/pull/19)
* More Actions Work by [@jak574](https://github.com/jak574) in [#20](https://github.com/CosmicFrontierLabs/rust-ephem/pull/20)
* More actions work...sigh by [@jak574](https://github.com/jak574) in [#21](https://github.com/CosmicFrontierLabs/rust-ephem/pull/21)
* Try maturin default build CI by [@jak574](https://github.com/jak574) in [#22](https://github.com/CosmicFrontierLabs/rust-ephem/pull/22)
* Final build CI updates.... by [@jak574](https://github.com/jak574) in [#23](https://github.com/CosmicFrontierLabs/rust-ephem/pull/23)


**Full Changelog**: https://github.com/CosmicFrontierLabs/rust-ephem/compare/v0.1.0...v0.1.1

[Changes][v0.1.1]


<a id="v0.1.0"></a>
## [v0.1.0: First release](https://github.com/CosmicFrontierLabs/rust-ephem/releases/tag/v0.1.0) - 2025-11-14

### What's Changed
* release: commit v0.1.0 of rust-ephem by [@jak574](https://github.com/jak574) in [#2](https://github.com/CosmicFrontierLabs/rust-ephem/pull/2)

## New Contributors
* [@jak574](https://github.com/jak574) made their first contribution in [#2](https://github.com/CosmicFrontierLabs/rust-ephem/pull/2)

**Full Changelog**: https://github.com/CosmicFrontierLabs/rust-ephem/commits/v0.1.0

[Changes][v0.1.0]


[v0.2.1]: https://github.com/CosmicFrontierLabs/rust-ephem/compare/v0.2.0...v0.2.1
[v0.2.0]: https://github.com/CosmicFrontierLabs/rust-ephem/compare/v0.1.15...v0.2.0
[v0.1.15]: https://github.com/CosmicFrontierLabs/rust-ephem/compare/v0.1.14...v0.1.15
[v0.1.14]: https://github.com/CosmicFrontierLabs/rust-ephem/compare/v0.1.13...v0.1.14
[v0.1.13]: https://github.com/CosmicFrontierLabs/rust-ephem/compare/v0.1.12...v0.1.13
[v0.1.12]: https://github.com/CosmicFrontierLabs/rust-ephem/compare/v0.1.11...v0.1.12
[v0.1.11]: https://github.com/CosmicFrontierLabs/rust-ephem/compare/v0.1.10...v0.1.11
[v0.1.10]: https://github.com/CosmicFrontierLabs/rust-ephem/compare/v0.1.9...v0.1.10
[v0.1.9]: https://github.com/CosmicFrontierLabs/rust-ephem/compare/v0.1.8...v0.1.9
[v0.1.8]: https://github.com/CosmicFrontierLabs/rust-ephem/compare/v0.1.7...v0.1.8
[v0.1.7]: https://github.com/CosmicFrontierLabs/rust-ephem/compare/v0.1.6...v0.1.7
[v0.1.6]: https://github.com/CosmicFrontierLabs/rust-ephem/compare/v0.1.5...v0.1.6
[v0.1.5]: https://github.com/CosmicFrontierLabs/rust-ephem/compare/v0.1.4...v0.1.5
[v0.1.4]: https://github.com/CosmicFrontierLabs/rust-ephem/compare/v0.1.3...v0.1.4
[v0.1.3]: https://github.com/CosmicFrontierLabs/rust-ephem/compare/v0.1.2...v0.1.3
[v0.1.2]: https://github.com/CosmicFrontierLabs/rust-ephem/compare/v0.1.1...v0.1.2
[v0.1.1]: https://github.com/CosmicFrontierLabs/rust-ephem/compare/v0.1.0...v0.1.1
[v0.1.0]: https://github.com/CosmicFrontierLabs/rust-ephem/tree/v0.1.0

<!-- Generated by https://github.com/rhysd/changelog-from-release v3.9.1 -->
