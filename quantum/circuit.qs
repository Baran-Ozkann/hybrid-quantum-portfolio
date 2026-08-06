// ============================================================================
// circuit.qs — Phase 3: Parametrized quantum circuit (hardware-efficient VQE)
// ============================================================================
//
// Canonical Q# source for the 4-qubit variational ansatz. Read from disk by
// quantum/q_bridge.py and evaluated via qsharp.eval(). Top-level declarations
// (no `namespace` block) so Python can call operations by bare name.
//
// ----------------------------------------------------------------------------
// QUBIT ROLES
// ----------------------------------------------------------------------------
// One qubit per asset. Qubit i carries the binary "hold / do-not-hold" decision
// for asset i, and its Pauli-Z expectation drives that asset's portfolio
// allocation (see docs/encoding_schema.md).
//
//   qubit 0 -> TSLA      qubit 2 -> AMD
//   qubit 1 -> NVDA      qubit 3 -> COIN
//
// Bit ordering: Q# is BIG-ENDIAN here -- qubit 0 is the MOST significant bit of
// the state-vector index. Verified empirically in Phase 1: Ry(pi) applied to
// qubit 3 alone drives the system to basis index 1 (binary 0001). Python-side
// decoding must therefore use  bit(q) = (index >> (numQubits - 1 - q)) & 1.
//
// ----------------------------------------------------------------------------
// ANSATZ STRUCTURE
// ----------------------------------------------------------------------------
// `numLayers` repetitions of:
//
//     Ry(theta) on every qubit          <- 4 params, populates amplitudes
//     Rz(phi)   on every qubit          <- 4 params, populates relative phases
//     CNOT ladder q0->q1->q2->q3        <- 0 params, creates entanglement
//
// followed by ONE final Ry layer:
//
//     Ry(theta) on every qubit          <- 4 params
//
// Total parameters = numQubits * (2 * numLayers + 1)
//                  = 4 * (2*3 + 1) = 28   for the default numLayers = 3.
//
// WHY THE TRAILING LAYER IS Ry AND NOT Rz
// ---------------------------------------
// The portfolio Hamiltonian is DIAGONAL in the computational (Z) basis, so
// observable outcomes depend only on amplitude MAGNITUDES |psi_k|. An Rz gate
// is diagonal and unit-modulus, and the CNOT ladder is a permutation matrix;
// neither changes |psi_k|. A trailing Rz layer would therefore be exactly
// inert -- its parameters would carry identically zero gradient and waste 4 of
// the 28 optimizer degrees of freedom. Ending on Ry keeps every parameter live.
// (Rz layers in the INTERIOR are NOT inert: the phases they write are converted
// into amplitude differences by the Ry gates of the following layer.)
//
// ----------------------------------------------------------------------------
// QUBIT-RELEASE CONTRACT
// ----------------------------------------------------------------------------
// Q# requires allocated qubits to be back in |0> at end of scope, else the
// runtime raises `Qsc.Eval.ReleasedQubitNotZero`. `ApplyAnsatz` deliberately
// does NOT allocate or reset -- it operates on a caller-owned register so the
// Python exact-expectation backend can inspect the live state with
// dump_machine() before resetting. The sampling operations own and reset their
// own registers.
// ============================================================================

import Std.Convert.IntAsDouble;
import Std.Diagnostics.Fact;
import Std.Measurement.MResetEachZ;

/// Apply the hardware-efficient ansatz IN PLACE to a caller-owned register.
///
/// Allocates nothing and resets nothing, so the caller may read the resulting
/// state vector (exact backend) or measure it (sampling backend).
///
/// # Input
/// ## qs
/// Caller-owned qubit register, expected in |0...0> on entry.
/// ## params
/// Flat rotation angles, length must equal Length(qs) * (2 * numLayers + 1).
/// Layout, with n = Length(qs), for layer L in 0..numLayers-1:
///     params[L*2n + 0  .. L*2n + n-1]  -> Ry angles for that layer
///     params[L*2n + n  .. L*2n + 2n-1] -> Rz angles for that layer
/// and the final n entries are the trailing Ry layer.
/// ## numLayers
/// Number of (Ry, Rz, CNOT-ladder) repetitions.
operation ApplyAnsatz(qs : Qubit[], params : Double[], numLayers : Int) : Unit {
    let n = Length(qs);
    let expected = n * (2 * numLayers + 1);
    Fact(
        Length(params) == expected,
        $"ApplyAnsatz: expected {expected} params for {n} qubits and "
        + $"{numLayers} layers, got {Length(params)}."
    );

    for layer in 0..numLayers - 1 {
        let offset = layer * 2 * n;

        // Amplitude rotations: real-valued mixing of |0> and |1>.
        for i in 0..n - 1 {
            Ry(params[offset + i], qs[i]);
        }

        // Phase rotations: relative phase between |0> and |1>.
        for i in 0..n - 1 {
            Rz(params[offset + n + i], qs[i]);
        }

        // Linear CNOT ladder: q0->q1, q1->q2, q2->q3. Nearest-neighbour only,
        // so the circuit maps onto hardware without SWAP overhead.
        for i in 0..n - 2 {
            CNOT(qs[i], qs[i + 1]);
        }
    }

    // Trailing Ry layer (see header for why this is Ry, not Rz).
    let finalOffset = numLayers * 2 * n;
    for i in 0..n - 1 {
        Ry(params[finalOffset + i], qs[i]);
    }
}

/// Number of ansatz parameters for a given register size and depth.
/// Mirrored by `n_params()` in q_bridge.py; kept here so the contract is
/// stated once in Q# as well.
function AnsatzParameterCount(numQubits : Int, numLayers : Int) : Int {
    return numQubits * (2 * numLayers + 1);
}

/// As `ApplyAnsatz`, but the CNOT ladder is CLOSED INTO A RING: one extra
/// CNOT(q[n-1], q[0]) per layer.
///
/// # Why this exists
/// The ladder is applied sequentially, so within a single layer influence
/// travels forward (low index -> high index) across the whole register, but
/// backward only one qubit per layer -- a CNOT writes to its target. Measured
/// on 12 qubits (`varweight/probe_lightcone.py`): backward reach equals the
/// layer count exactly, 1/2/3/6 at depths 1/2/3/6, against the 11 needed to
/// span. At 4 qubits, 3 layers spans the register and the asymmetry never
/// bites; at 12 it does, and it produced a weight inversion between the two
/// assets at opposite ends of the register.
///
/// Closing the ring lets forward flow wrap around, so backward reach becomes
/// full in one or two layers. It costs ONE GATE PER LAYER and ZERO extra
/// parameters -- the parameter count is identical to `ApplyAnsatz`, so the two
/// are drop-in alternatives at the same depth.
///
/// # Note
/// `ApplyAnsatz` is deliberately left untouched: every Phase 1-9 result was
/// measured on it, and those results stand as measured.
operation ApplyAnsatzRing(qs : Qubit[], params : Double[], numLayers : Int) : Unit {
    let n = Length(qs);
    let expected = n * (2 * numLayers + 1);
    Fact(
        Length(params) == expected,
        $"ApplyAnsatzRing: expected {expected} params for {n} qubits and "
        + $"{numLayers} layers, got {Length(params)}."
    );

    for layer in 0..numLayers - 1 {
        let offset = layer * 2 * n;

        for i in 0..n - 1 {
            Ry(params[offset + i], qs[i]);
        }
        for i in 0..n - 1 {
            Rz(params[offset + n + i], qs[i]);
        }
        for i in 0..n - 2 {
            CNOT(qs[i], qs[i + 1]);
        }
        // The closure. Guarded because at n = 2 it would duplicate the ladder's
        // only CNOT, and at n < 2 there is nothing to entangle.
        if n > 2 {
            CNOT(qs[n - 1], qs[0]);
        }
    }

    let finalOffset = numLayers * 2 * n;
    for i in 0..n - 1 {
        Ry(params[finalOffset + i], qs[i]);
    }
}

/// Estimate <H> for the diagonal Ising portfolio Hamiltonian by SAMPLING.
///
/// Because H is diagonal in the Z basis, every term commutes and a single
/// computational-basis measurement per shot suffices -- no basis-rotation
/// passes are required.
///
///     H = sum_i h_i * s_i  +  sum_{i<j} J_ij * s_i * s_j,   s in {+1,-1}
///
/// with the standard spin convention  Zero -> s = +1,  One -> s = -1.
///
/// # Input
/// ## params
/// Ansatz angles (see `ApplyAnsatz`).
/// ## hCoeffs
/// Linear coefficients [h_0, h_1, h_2, h_3].
/// ## jCoeffs
/// Quadratic coefficients in ROW-MAJOR UPPER-TRIANGULAR order:
/// [J_01, J_02, J_03, J_12, J_13, J_23].
/// ## numLayers
/// Ansatz depth.
/// ## numShots
/// Measurement shots to average over. Statistical error on the returned mean
/// falls as 1/sqrt(numShots).
///
/// Returns: the sample mean of the Ising energy, an estimator of <H>.
operation EvaluateHamiltonian(
    params : Double[],
    hCoeffs : Double[],
    jCoeffs : Double[],
    numLayers : Int,
    numShots : Int
) : Double {
    let numQubits = Length(hCoeffs);
    mutable totalEnergy = 0.0;

    for _ in 0..numShots - 1 {
        use qs = Qubit[numQubits];
        ApplyAnsatz(qs, params, numLayers);

        // MResetEachZ measures every qubit in Z and returns the register to
        // |0...0>, satisfying the release contract in one step.
        let results = MResetEachZ(qs);

        // Result -> Ising spin.
        mutable spins = [0.0, size = numQubits];
        for i in 0..numQubits - 1 {
            set spins w/= i <- (results[i] == Zero ? 1.0 | -1.0);
        }

        mutable energy = 0.0;
        for i in 0..numQubits - 1 {
            set energy += hCoeffs[i] * spins[i];
        }
        mutable jIdx = 0;
        for i in 0..numQubits - 2 {
            for j in i + 1..numQubits - 1 {
                set energy += jCoeffs[jIdx] * spins[i] * spins[j];
                set jIdx += 1;
            }
        }

        set totalEnergy += energy;
    }

    return totalEnergy / IntAsDouble(numShots);
}

/// Sample computational-basis bitstrings from the ansatz.
///
/// Used by Phase 6 to decode measurement outcomes into portfolio weights, and
/// as an independent cross-check that the sampled distribution matches the
/// exact state vector read out by the Python backend.
///
/// # Input
/// ## params
/// Ansatz angles (see `ApplyAnsatz`).
/// ## numQubits
/// Register size (4 for this project).
/// ## numLayers
/// Ansatz depth.
/// ## numShots
/// Number of bitstrings to draw.
///
/// Returns: `numShots` arrays of length `numQubits`, entries in {0, 1}, where
/// element i is the outcome for qubit i (qubit 0 first).
operation MeasurePortfolio(
    params : Double[],
    numQubits : Int,
    numLayers : Int,
    numShots : Int
) : Int[][] {
    mutable allResults = [[0, size = numQubits], size = 0];

    for _ in 0..numShots - 1 {
        use qs = Qubit[numQubits];
        ApplyAnsatz(qs, params, numLayers);
        let results = MResetEachZ(qs);

        mutable bitstring = [0, size = numQubits];
        for i in 0..numQubits - 1 {
            set bitstring w/= i <- (results[i] == One ? 1 | 0);
        }
        set allResults += [bitstring];
    }

    return allResults;
}
