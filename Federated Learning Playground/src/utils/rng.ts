// rng.ts
import seedrandom from 'seedrandom';

declare global {
  interface Math {
    seed(seed?: number): seedrandom.prng;
  }
}

export {};

Math.seed = function (seed?: number) {
  // Make deterministic RNG and override Math.random()
  const prng = seedrandom(String(seed));
  return prng;
};

declare let ga: any;