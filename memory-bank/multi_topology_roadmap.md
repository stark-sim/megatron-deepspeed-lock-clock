# Multi-Topology Extension Roadmap

## Overview

This document outlines the systematic extension of the network-aware predictor to 
diverse network topologies, parallelism strategies, and hardware configurations.

## Tier-1: Network Layer Extensions

### 1.1 RoCE (RDMA over Converged Ethernet)
**Priority**: High  
**Timeline**: Q2 2025

| Bandwidth | Expected α_dp | Test Environment |
|-----------|---------------|------------------|
| 25 Gbps   | 2.0×10⁻¹¹     | Cloud providers  |
| 50 Gbps   | 1.0×10⁻¹¹     | Enterprise DC    |
| 100 Gbps  | 5.0×10⁻¹²     | HPC clusters     |

**Implementation**:
- Run `torch_nccl_comm_bench.py` on RoCE-enabled cluster
- Characterize bandwidth/latency/jitter tradeoffs
- Extend `fit_cross_node_penalty_model()` with RoCE profiles

**Validation**: 2×4 → 2×8 transfer on RoCE cluster

### 1.2 Ethernet (Standard TCP/IP)
**Priority**: Medium  
**Timeline**: Q2-Q3 2025

| Bandwidth | Expected α_dp | Use Case |
|-----------|---------------|----------|
| 1 Gbps    | 4.2×10⁻¹⁰     | Development  |
| 10 Gbps   | 4.2×10⁻¹¹     | Small clusters |

**Challenges**:
- Higher latency variance
- CPU overhead for protocol processing
- Need larger warmup iterations for stable measurements

### 1.3 Shared HPC Interconnects
**Priority**: Medium  
**Timeline**: Q3 2025

**Scenarios**:
- Multiple jobs sharing IB fabric
- Quality-of-Service (QoS) constraints
- Adaptive routing effects

**Approach**:
- Measure bandwidth under contention
- Model contention-aware penalties
- Runtime adaptation based on observed bandwidth

## Tier-2: Topology Layer Extensions

### 2.1 Tensor Parallelism Scaling
**Priority**: High  
**Timeline**: Q2 2025

| Configuration | Cross-Node TP | Expected Overhead |
|---------------|---------------|-------------------|
| TP=2, PP=2   | Yes           | Low (NVLink pref) |
| TP=4, PP=1   | No            | Minimal           |
| TP=8, PP=1   | Yes           | High              |

**Key Insight**: TP across nodes is generally avoided (latency-sensitive), but 
may be necessary for large models.

### 2.2 Pipeline Parallelism Depth
**Priority**: High  
**Timeline**: Q2 2025

| Configuration | PP Stages | Cross-Node Edges |
|---------------|-----------|------------------|
| TP=2, PP=4   | 4         | 2                |
| TP=2, PP=8   | 8         | 4                |
| TP=1, PP=8   | 8         | 4                |

**Model Extensions**:
- Pipeline bubble modeling
- Activation compression effects
- Microbatch scheduling impact

### 2.3 Data Parallelism Scale
**Priority**: High  
**Timeline**: Ongoing

Already validated:
- ✓ DP=2 (2×4 configuration)
- ✓ DP=4 (2×8 configuration)

Planned:
- DP=8 (2×16 configuration)
- DP=16 (4×8 configuration)

### 2.4 3D Parallelism (TP + PP + DP)
**Priority**: Medium  
**Timeline**: Q3 2025

**Configurations**:
- TP=2, PP=2, DP=4 (2×8)
- TP=2, PP=4, DP=2 (2×8)
- TP=4, PP=2, DP=2 (2×8)

**Challenge**: Interaction effects between parallelism dimensions

## Tier-3: Hardware Layer Extensions

### 3.1 NVIDIA A100
**Priority**: High  
**Timeline**: Q3 2025

| Aspect | V100 | A100 | Impact |
|--------|------|------|--------|
| NVLink | Gen 2 | Gen 3 | 2× bandwidth |
| Compute | 125 TFLOPS | 312 TFLOPS | Different saturation |
| Memory | 32 GB HBM2 | 40/80 GB HBM2e | Larger models |

**Adaptations Needed**:
- Update hardware specifications
- Recalibrate compute/memory limits
- Characterize A100-specific power curves

### 3.2 NVIDIA H100
**Priority**: Medium  
**Timeline**: Q4 2025

| Aspect | H100 | Impact |
|--------|------|--------|
| NVLink | Gen 4 | 3.5× bandwidth vs V100 |
| Compute | 989 TFLOPS | Compute-bound shifts |
| Transformer Engine | Hardware acceleration | New modeling dimension |

### 3.3 AMD MI200/MI300
**Priority**: Low  
**Timeline**: 2026

**Considerations**:
- Different interconnect (Infinity Fabric vs NVLink)
- ROCm software stack differences
- Power/performance characteristics

### 3.4 Multi-Node GPU Topologies
**Priority**: Medium  
**Timeline**: Q4 2025

| Topology | Nodes × GPUs | Network |
|----------|--------------|---------|
| 4×8      | 32           | IB/Fat-tree |
| 8×8      | 64           | IB/Fat-tree |
| 2×16     | 32           | IB/Rail-optimized |

**Challenges**:
- Hierarchical communication patterns
- Network topology-aware placement
- Collective algorithm selection (ring vs tree)

## Implementation Strategy

### Phase 1: Foundation (Current)
- ✅ V100 + IB validation
- ✅ 2×4 → 2×8 transfer
- ✅ Dynamic network detection

### Phase 2: Network Diversity (Q2 2025)
- [ ] RoCE 25/50/100 Gbps validation
- [ ] Ethernet 1/10 Gbps characterization
- [ ] Continuous penalty function development

### Phase 3: Topology Diversity (Q2-Q3 2025)
- [ ] TP=2, PP=2, DP=4 validation
- [ ] TP=2, PP=4, DP=2 validation
- [ ] 2×16 (DP=8) scaling validation

### Phase 4: Hardware Diversity (Q3-Q4 2025)
- [ ] A100 platform validation
- [ ] H100 platform validation
- [ ] Cross-hardware transfer capabilities

### Phase 5: Large Scale (2026)
- [ ] 4×8 (32 GPU) validation
- [ ] 8×8 (64 GPU) validation
- [ ] Production deployment support

## Validation Framework

### For Each New Environment
1. **Network Benchmark**: Run `torch_nccl_comm_bench.py`
2. **Source Calibration**: Collect 2-3 frequency points on base topology
3. **Target Transfer**: Validate on target topology
4. **Accuracy Threshold**: MAPE < 15% for acceptance
5. **Documentation**: Update `experimental_data.md`

### Test Matrix

| Network | GPUs | Topology | Status |
|---------|------|----------|--------|
| IB 100G | 2×4  | TP4PP1DP2 | ✅ Validated |
| IB 100G | 2×8  | TP4PP1DP4 | ✅ Validated |
| RoCE 50G| 2×4  | TP4PP1DP2 | 📝 Planned |
| RoCE 50G| 2×8  | TP4PP1DP4 | 📝 Planned |
| Eth 10G | 2×4  | TP4PP1DP2 | 📝 Planned |
| IB 100G | 2×16 | TP4PP1DP8 | 📝 Planned |
| IB 100G | 4×8  | TP4PP1DP8 | 📝 Planned |

## Key Metrics

### Prediction Accuracy
- Time MAPE: < 10% (excellent), < 20% (acceptable)
- Power MAPE: < 10% (excellent), < 15% (acceptable)
- Energy MAPE: < 15% (excellent), < 25% (acceptable)

### Calibration Efficiency
- Source points: 2-3 frequencies sufficient
- Benchmark time: < 60 seconds
- Calibration time: < 5 minutes

### Generalization
- Cross-topology: Different TP/PP/DP combinations
- Cross-network: Same hardware, different interconnect
- Cross-hardware: Different GPU generations

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| A100 access delays | Medium | Schedule slip | Partner with cloud providers |
| RoCE instability | Medium | Validation noise | Multiple measurement rounds |
| Large-scale OOM | High | Testing blockage | Gradual memory scaling |
| Network contention | Medium | Measurement bias | Off-peak testing windows |

## Resources Needed

### Hardware Access
- RoCE cluster: 2-4 nodes with 25/50Gbps NICs
- A100 cluster: 2-4 nodes with A100 GPUs
- H100 cluster: 2-4 nodes with H100 GPUs (optional)

### Compute Budget
- Network characterization: ~100 GPU-hours per environment
- Full validation: ~500 GPU-hours per tier
- Total estimate: ~5,000 GPU-hours for complete roadmap

### Personnel
- 1 FTE: Core development and validation
- 0.5 FTE: Paper writing and documentation
- 0.25 FTE: Large-scale testing coordination

## Success Criteria

By end of 2025:
- [ ] 5+ network environments validated
- [ ] 10+ topology configurations tested
- [ ] 2+ hardware platforms supported
- [ ] 1 production deployment reference
- [ ] Paper accepted at top-tier venue
