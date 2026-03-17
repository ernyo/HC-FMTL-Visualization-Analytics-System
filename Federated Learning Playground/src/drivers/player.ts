import * as d3 from 'd3';

export interface Driver {
  step: () => Promise<boolean>;
}

export class PlayerDriver implements Driver {
  constructor(private deps: {oneStep: (epochsPerClient: number) => Promise<void>;}) {}

  async step() {
    try {
      const epochs = Number(d3.select("#epochs-per-client").property("value"));
      await this.deps.oneStep(epochs);
      return true;
    } catch (err) {
      console.error("Error during training step:", err);
      return false;
    }
  }
}

// === Training loop control
export class Player {
  private timerIndex = 0;
  private isPlaying = false;
  private callback: (isPlaying: boolean) => void = null;

  constructor(private driver: Driver) {}

  setDriver(driver: Driver) {
    this.driver = driver;
  }

  /** Plays/pauses the player. */
  playOrPause() {
    if (this.isPlaying) {
      this.isPlaying = false;
      this.pause();
    } else {
      this.isPlaying = true;
      this.play();
    }
  }

  onPlayPause(callback: (isPlaying: boolean) => void) {
    this.callback = callback;
  }

  play() {
    this.pause();
    this.isPlaying = true;
    if (this.callback) {
      this.callback(this.isPlaying);
    }
    this.start(this.timerIndex);
  }

  pause() {
    this.timerIndex++;
    this.isPlaying = false;
    if (this.callback) {
      this.callback(this.isPlaying);
    }
  }

  private stepping = false;

  private start(localTimerIndex: number) {
    d3.timer(() => {
      // cancel old timers
      if (localTimerIndex < this.timerIndex) return true;

      // prevent overlap
      if (this.stepping) return false;

      this.stepping = true;

      // run async work, but don't make the timer callback async
      void (async () => {
        try {
          const keepGoing = await this.driver.step();
          if (!keepGoing) this.pause(); // pause will bump timerIndex / stop future ticks
        } finally {
          this.stepping = false;
        }
      })();

      return false; // keep timer alive
    }, 0);
  }
}