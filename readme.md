# Differentiable Evolution Tree Search

A JAX-based implementation of differentiable algorithms for phylogenetic tree search, providing a modern alternative to traditional heuristic methods through gradient-based optimization.

## Overview

This project explores differentiable phylogenetic algorithms that leverage the JAX ecosystem for high-performance, JIT-compiled tree search operations. By implementing classic phylogenetic algorithms (such as Sankoff's algorithm for maximum parsimony) in a differentiable framework, we enable gradient-based optimization of both tree topologies and ancestral sequences.

The core innovation lies in making traditionally discrete phylogenetic operations differentiable while maintaining computational efficiency through JAX's functional programming paradigm and automatic differentiation capabilities.

## Key Features & Differentiable Components

### Differentiable Algorithms

- **Sankoff's Algorithm**: Differentiable implementation for maximum parsimony scoring
- **Tree Topology Optimization**: Gradient-based search over tree space
- **Ancestral Sequence Reconstruction**: Differentiable inference of internal node sequences
- **Batch Processing**: Vectorized operations for multiple tree evaluations

### JAX Integration

- **JIT Compilation**: All core algorithms are JIT-compatible for optimal performance
- **Functional Programming**: Immutable data structures and pure functions throughout
- **Automatic Differentiation**: Full gradient computation for all model parameters
- **VMAP Support**: Efficient batching across multiple trees or datasets

## Getting Started

### Installation

1. **Clone the repository:**

   ```bash
   git clone <repository-url>
   cd trex
   ```markdown

2. **Install dependencies:**

   ```bash
   uv sync --extra cuda  # For GPU
   uv sync --extra tpu   # For TPU
   uv sync --extra cpu   # For CPU-only (default)
   ```

## Testing

### Unit Tests

Run the complete test suite:

```bash
python -m pytest tests/
```

### Convergence Tests

Verify algorithmic correctness and convergence properties:

```bash
python -m pytest tests/test_convergence.py -v
```

The test suite covers:

- Numerical correctness of differentiable algorithms
- Gradient computation accuracy
- JAX transformation compatibility (`jit`, `vmap`, `scan`)
- Edge cases and boundary conditions
- Performance benchmarks

### Test Coverage

Generate coverage reports:

```bash
python -m pytest tests/ --cov=modules --cov=src/trex --cov-report=html
```

## Code Structure

```bash
trex/
├── modules/                    # Core algorithmic components
│   ├── sankoff.py             # Differentiable Sankoff algorithm
│   └── ...                    # Additional phylogenetic modules
├── src/trex/                  # Main package source
├── tests/                     # Comprehensive test suite
│   ├── test_convergence.py    # Convergence and accuracy tests
│   └── ...                    # Unit tests mirroring module structure
├── pyproject.toml             # Project configuration and dependencies
├── AGENTS.md                  # Development guidelines and coding standards
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

### Key Directories

- **`modules/`**: Contains the core differentiable phylogenetic algorithms, each implemented as pure JAX functions
- **`src/trex/`**: Main package organization following Python packaging best practices
- **`tests/`**: Comprehensive test coverage including unit tests and convergence validation

### Configuration Files

- **`pyproject.toml`**: Project metadata, dependencies, and tool configurations (Ruff, Pyright)
- **`AGENTS.md`**: Detailed coding standards and development practices for contributors

## Development Standards

This project adheres to strict code quality standards:

- **Linting**: Ruff with comprehensive rule enforcement (`select = ["ALL"]`)
- **Type Checking**: Pyright in strict mode for complete type safety
- **Testing**: High coverage requirements (target: 90%) using pytest
- **Documentation**: Google-style docstrings for all public functions
- **JAX Compatibility**: All numerical code must be JIT/VMAP/SCAN compatible

## Current Limitations & Planned Improvements

### Known Limitations

- **Tree Topology Representation**: Current implementation may have constraints on tree topology parameterization
- **Scalability**: Performance evaluation needed for very large phylogenies (>100 taxa)
- **Algorithm Coverage**: Limited to maximum parsimony; maximum likelihood methods planned

### Future Developments

- **Maximum Likelihood**: Differentiable implementation of likelihood-based inference
- **Advanced Tree Moves**: More sophisticated topology proposal mechanisms
- **Benchmarking Suite**: Comprehensive comparison with traditional phylogenetic software
- **GPU Acceleration**: Full exploitation of JAX's GPU capabilities for large-scale problems
- **Bayesian Methods**: Integration of probabilistic programming for uncertainty quantification

### Future Research Directions

- **Hybrid Methods**: Combining gradient-based optimization with traditional heuristics
- **Neural Architectures**: Integration with neural network components for learned priors
- **Multi-objective Optimization**: Simultaneous optimization of multiple phylogenetic criteria

## Contribution Guidelines

Please refer to `AGENTS.md` for detailed contribution guidelines, including:

- JAX programming best practices
- Code style and linting requirements
- Testing standards and procedures
- Documentation expectations

## License

[License information to be added]

## Citation

[Citation information to be added upon publication]

- **Testing**: High coverage requirements (target: 90%) using pytest
- **Documentation**: Google-style docstrings for all public functions
- **JAX Compatibility**: All numerical code must be JIT/VMAP/SCAN compatible

## Current Limitations & Future Work

### Limitations

- **Tree Topology Representation**: Current implementation may have constraints on tree topology parameterization
- **Scalability**: Performance evaluation needed for very large phylogenies (>100 taxa)
- **Algorithm Coverage**: Limited to maximum parsimony; maximum likelihood methods planned

### Planned Developments

- **Maximum Likelihood**: Differentiable implementation of likelihood-based inference
- **Advanced Tree Moves**: More sophisticated topology proposal mechanisms
- **Benchmarking Suite**: Comprehensive comparison with traditional phylogenetic software
- **GPU Acceleration**: Full exploitation of JAX's GPU capabilities for large-scale problems
- **Bayesian Methods**: Integration of probabilistic programming for uncertainty quantification

### Research Directions

- **Hybrid Methods**: Combining gradient-based optimization with traditional heuristics
- **Neural Architectures**: Integration with neural network components for learned priors
- **Multi-objective Optimization**: Simultaneous optimization of multiple phylogenetic criteria

## Contributing

Please refer to `AGENTS.md` for detailed contribution guidelines, including:

- JAX programming best practices
- Code style and linting requirements
- Testing standards and procedures
- Documentation expectations

## License Details

[License information to be added]

## Citation Details

[Citation information to be added upon publication]
