// Render the IS-ratio formulas of the README as SVG files (Computer Modern fonts).
//
// Setup:  npm install mathjax-full
// Usage:  NODE_PATH=$(pwd)/node_modules node make_is_ratio_formulas.js
// Output: docs/verl_dllm_agent/formulas/<name>_{light,dark}.svg

const fs = require("fs");
const path = require("path");
const { mathjax } = require("mathjax-full/js/mathjax.js");
const { TeX } = require("mathjax-full/js/input/tex.js");
const { SVG } = require("mathjax-full/js/output/svg.js");
const { liteAdaptor } = require("mathjax-full/js/adaptors/liteAdaptor.js");
const { RegisterHTMLHandler } = require("mathjax-full/js/handlers/html.js");
const { AllPackages } = require("mathjax-full/js/input/tex/AllPackages.js");

const FORMULAS = {
  ell: String.raw`\ell_{j,k}(\theta)=\frac{\mathbf{1}[\,j\in\mathcal{M}_k\,]}{p_{j,k}}\,\log\pi_\theta\left(z_j\mid u_k\right)`,
  delta: String.raw`\Delta_j=\sum_{k=1}^{K}\alpha_k\left[\ell_{j,k}(\theta)-\ell_{j,k}(\theta_{\mathrm{old}})\right]`,
  token: String.raw`\rho_i=\exp\left(\Delta_i\right)`,
  block: String.raw`\rho_b=\exp\left(\frac{1}{B}\sum_{j\in b}\Delta_j\right)`,
  sequence: String.raw`\rho_{\mathrm{seq}}=\exp\left(\frac{1}{L}\sum_{j=1}^{|z|}\Delta_j\right)`,
};
const THEMES = { light: "#1f2328", dark: "#e6edf3" };
const SCALE = 1.25;

const adaptor = liteAdaptor();
RegisterHTMLHandler(adaptor);
const doc = mathjax.document("", {
  InputJax: new TeX({ packages: AllPackages }),
  OutputJax: new SVG({ fontCache: "none" }),
});

const outDir = path.join(__dirname, "formulas");
fs.mkdirSync(outDir, { recursive: true });

for (const [name, tex] of Object.entries(FORMULAS)) {
  const node = doc.convert("\\displaystyle " + tex, { display: true });
  let svg = adaptor.innerHTML(node);
  svg = svg.replace(/(width|height)="([\d.]+)ex"/g, (_, k, v) => `${k}="${(v * SCALE).toFixed(3)}ex"`);
  for (const [theme, color] of Object.entries(THEMES)) {
    const colored = svg.replace(/currentColor/g, color);
    fs.writeFileSync(path.join(outDir, `${name}_${theme}.svg`), colored + "\n");
  }
}
