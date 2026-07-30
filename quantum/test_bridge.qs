// ============================================================================
// test_bridge.qs — Phase 1: Python <-> Q# interoperability validation
// ============================================================================
//
// This file is the CANONICAL source for the bridge-test operations. It is read
// from disk by quantum/q_bridge.py and handed to qsharp.eval(). It is NOT
// compiled as a standalone Q# project.
//
// Style note: operations are declared at top level WITHOUT a `namespace` block.
// The qdk 1.30.0 interpreter (qsharp.eval) evaluates source into the current
// session scope, so top-level declarations are directly callable by bare name
// (e.g. "TestBridge()"). A `namespace` wrapper would require qualified names.
//
// Qubit-release contract: Q# requires every allocated qubit to be returned to
// |0> before it leaves scope, otherwise the runtime raises
// `Qsc.Eval.ReleasedQubitNotZero`. Every operation below resets what it uses.
// ============================================================================

/// Allocate one qubit, apply an X (bit-flip) gate, and measure it.
///
/// Since X|0> = |1>, a noiseless simulator must return `One` every time. This
/// is the minimal end-to-end proof that Python can drive Q# and read back a
/// classical result.
///
/// Returns: `One` on a correct, noiseless run.
operation TestBridge() : Result {
    use q = Qubit();
    X(q);
    let result = M(q);
    Reset(q);                 // restore |0> before release
    return result;
}

/// Control test for TestBridge: measure an untouched qubit.
///
/// Confirms the simulator is not simply returning `One` unconditionally --
/// without this, TestBridge passing would be uninformative.
///
/// Returns: `Zero` on a correct, noiseless run.
operation TestIdentity() : Result {
    use q = Qubit();
    let result = M(q);
    Reset(q);
    return result;
}

/// Validate Double[] marshalling across the Python -> Q# boundary.
///
/// Python formats a float list into a Q# array literal; Q# sums it and returns
/// a Double. Agreement with Python's own sum proves values survive the round
/// trip without truncation or reordering.
///
/// # Input
/// ## values
/// Array of doubles supplied by Python.
///
/// Returns: the arithmetic sum of `values`.
operation TestParameterPassing(values : Double[]) : Double {
    mutable total = 0.0;
    for v in values {
        set total += v;
    }
    return total;
}

/// Validate Int -> Q# marshalling and multi-qubit allocation.
///
/// Allocates `n` qubits, flips all of them, measures, and returns the number
/// of `One` outcomes -- which must equal `n`. This exercises the array
/// allocation path used by the Phase 3 ansatz.
///
/// # Input
/// ## n
/// Number of qubits to allocate (expected: 4 for this project).
///
/// Returns: count of qubits measured as `One` (expected: `n`).
operation TestMultiQubit(n : Int) : Int {
    use qs = Qubit[n];
    for q in qs {
        X(q);
    }
    mutable ones = 0;
    for q in qs {
        if M(q) == One {
            set ones += 1;
        }
    }
    ResetAll(qs);
    return ones;
}
