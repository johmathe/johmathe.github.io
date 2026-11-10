/** A mounted figure. `apply` puts it in the state a numbered caption step describes. */
export interface Figure {
  apply?(step: string): void;
}

/** Builds a figure into `root`, replacing its loading text. */
export type Mount = (root: HTMLElement) => Promise<Figure>;
