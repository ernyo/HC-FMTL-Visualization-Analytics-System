import * as tf from "@tensorflow/tfjs";
import { describe, it, expect, vi, beforeAll, afterEach } from "vitest";

vi.mock("../federated/utils", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../federated/utils")>();
  return {
    ...actual,
    solveSimplexProjectedGD: vi.fn(), // override only this
  };
});

// Now import after the mock
import { getCaDelta } from "../federated/aggregate";
import { solveSimplexProjectedGD, EPS } from "../federated/utils";

beforeAll(async () => {
  await tf.setBackend("cpu");
  await tf.ready();
});

afterEach(() => {
  vi.clearAllMocks();
});

function expectAllClose(a: tf.Tensor, b: tf.Tensor, atol = 1e-5, rtol = 1e-5) {
  const aArr = a.arraySync() as number[];
  const bArr = b.arraySync() as number[];
  expect(aArr.length).toBe(bArr.length);

  for (let i = 0; i < aArr.length; i++) {
    const diff = Math.abs(aArr[i] - bArr[i]);
    const thresh = atol + rtol * Math.abs(bArr[i]);
    expect(diff).toBeLessThanOrEqual(thresh);
  }
}

describe("getCaDelta post-solver formula", () => {
  it("matches manual reconstruction with mocked weights", () => {
    const d1 = tf.tensor1d([1, 2, 3], "float32");
    const d2 = tf.tensor1d([-1, 0.5, 4], "float32");
    const deltas = [d1, d2];
    const alpha = 0.7;

    const wKnown = new Float64Array([0.25, 0.75]);
    (solveSimplexProjectedGD as unknown as ReturnType<typeof vi.fn>).mockReturnValue(wKnown);

    const got = getCaDelta(deltas, alpha);

    const expected = tf.tidy(() => {
      const N = deltas.length;

      const grads = tf.stack(deltas).transpose();      // [d, N]
      const GG = grads.transpose().matMul(grads);      // [N, N]
      const g0Norm = tf.sqrt(GG.mean().add(EPS));
      const c = alpha * (g0Norm.arraySync() as number) + EPS;

      const ww = tf.tensor1d(Array.from(wKnown), "float32"); // [N]
      const gw = grads.mul(ww.reshape([1, N])).sum(1);       // [d]
      const lambda = tf.scalar(c).div(tf.norm(gw).add(EPS));
      const g = grads.mean(1).add(gw.mul(lambda));
      return g.div(tf.scalar(1 + alpha * alpha));
    });

    expectAllClose(got, expected);

    got.dispose();
    expected.dispose();
    d1.dispose();
    d2.dispose();
  });
});