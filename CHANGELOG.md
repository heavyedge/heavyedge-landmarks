# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.3.0] - 2025-11-09

### Added

- `minmax()` function for within-sample min-max scaling is introduced.

## [1.2.0] - 2025-11-05

### Added

- `preshape_dual()` function, which maps pre-shapes to configuration matrix space.

### Deprecated

- `dual_preshape()` function is deprecated. Use `preshape_dual()` function instead.

## [1.1.0] - 2025-10-28

### Added

- `preshape()` function, which transforms configuration matrices to pre-shapes.
- `dual_preshape()` function, which transforms configuration matrices to pre-shapes and then maps to the original space.

## [1.0.0] - 2025-10-14

### Added

- `pseudo_landmarks()` function.
- `landmarks_type2()` and `landmarks_type3()` functions.
- `plateau_type2()` and `plateau_type3()` functions.
