// One of the landing page's two heroes: condensation on a pane of glass.
//
// The pane holds a field of droplets that spell the wordmark, x0. Every pixel
// follows the forward and reverse process of a cosine-schedule diffusion model,
// x_t = cos(pi t / 2) x0 + sin(pi t / 2) eps, on its own clock t: the page opens
// fogged at t = 1 and every pixel steps back to t = 0, so fog clears into the
// droplets. The pointer breathes on the glass, sending the pixels near it
// forward again, and scrolling past fogs it all. The renderer reads x_t as the
// height of water on the glass: noise scatters light like fog, clean drops
// refract the lights behind the pane.

import {
	FULLSCREEN_VERTEX,
	GLSL_NOISE,
	bindTexture,
	createContext,
	createProgram,
	createTarget,
	createTexture,
	destroyTarget,
	drawFullscreen,
	fitCanvas,
	prefersReducedMotion,
	runLoop,
	type Target,
} from './gl';

const DURATION = 2.9; // seconds from t = 1 to t = 0 at the base rate; pixels run 0.62 to 2.1 times as fast
const SLOWEST = 0.62; // the slowest pixel's rate, relative to the base rate
const SIM_SCALE = 2; // CSS pixels per simulated pixel
const TIME_SCALE = 4; // simulated pixels per time-field cell

const T_UPDATE = `#version 300 es
precision highp float;
precision highp int;
in vec2 vUv;
uniform sampler2D uPrev;
uniform sampler2D uX0;
uniform float uDecay;
uniform float uFloor;
uniform vec4 uStroke;
uniform float uRadius;
uniform float uStrength;
uniform float uAspect;
out vec4 outColor;
float segment(vec2 p, vec2 a, vec2 b) {
	vec2 pa = p - a, ba = b - a;
	float k = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-9), 0.0, 1.0);
	return length(pa - ba * k);
}
${GLSL_NOISE}
// Smooth value noise in [0, 1], so neighbouring pixels keep similar clocks.
float valueNoise(vec2 q) {
	vec2 i = floor(q), f = fract(q);
	vec2 u = f * f * (3.0 - 2.0 * f);
	float a = hash_f(uvec3(uvec2(ivec2(i)), 7u));
	float b = hash_f(uvec3(uvec2(ivec2(i + vec2(1.0, 0.0))), 7u));
	float c = hash_f(uvec3(uvec2(ivec2(i + vec2(0.0, 1.0))), 7u));
	float d = hash_f(uvec3(uvec2(ivec2(i + vec2(1.0, 1.0))), 7u));
	return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}
void main() {
	// Every pixel walks the same schedule on its own clock: the letters clear first,
	// the rest of the pane unevenly after them.
	float letter = texture(uX0, vUv).g;
	float rate = mix(${SLOWEST.toFixed(2)}, 1.25, valueNoise(vUv * vec2(uAspect, 1.0) * 5.0)) * (1.0 + 0.7 * letter);
	float t = max(texture(uPrev, vUv).r - uDecay * rate, 0.0);
	vec2 s = vec2(uAspect, 1.0);
	float d = segment(vUv * s, uStroke.xy * s, uStroke.zw * s);
	float breath = uStrength * exp(-(d * d) / (uRadius * uRadius));
	outColor = vec4(max(max(t, breath), uFloor), 0.0, 0.0, 1.0);
}`;

const X_PASS = `#version 300 es
precision highp float;
precision highp int;
in vec2 vUv;
uniform sampler2D uX0;
uniform sampler2D uT;
uniform uint uSeedA;
uniform uint uSeedB;
uniform float uMix;
out vec4 outColor;
${GLSL_NOISE}
void main() {
	ivec2 p = ivec2(gl_FragCoord.xy);
	float t = texture(uT, vUv).r;
	float alpha = cos(1.5707963 * t);
	float sigma = sin(1.5707963 * t);
	float x0 = texelFetch(uX0, p, 0).r;
	// Two independent noise fields, rotated into each other, keep eps standard normal while it moves.
	float eps = cos(uMix) * gauss(p, uSeedA) + sin(uMix) * gauss(p, uSeedB);
	outColor = vec4(alpha * x0 + sigma * eps, 0.0, 0.0, 1.0);
}`;

const RENDER = `#version 300 es
precision highp float;
in vec2 vUv;
uniform sampler2D uX;
uniform sampler2D uT;
uniform vec2 uSim;
uniform vec2 uRes;
uniform float uTime;
// The palette, one per color theme (see PALETTES).
uniform vec3 uBase;
uniform vec3 uRise;
uniform vec3 uLights[4];
uniform float uLens;
uniform float uRim;
uniform vec3 uSpec;
uniform vec3 uBounce;
uniform vec3 uFog;
uniform float uFogKeep;
uniform float uVignette;
uniform float uGamma;
out vec4 outColor;

vec3 light(vec2 q, vec2 c, float r, vec3 color) {
	vec2 d = q - c;
	return color * exp(-dot(d, d) / (r * r));
}

// What lies behind the glass: a wash of color, with soft lights far away.
vec3 scene(vec2 q) {
	float aspect = uRes.x / uRes.y;
	vec3 col = uBase + uRise * q.y;
	col += light(q, vec2(aspect * 0.62 + 0.04 * sin(uTime * 0.05), 0.60), 0.36, uLights[0]);
	col += light(q, vec2(aspect * 0.36, 0.86 + 0.02 * cos(uTime * 0.04)), 0.28, uLights[1]);
	col += light(q, vec2(aspect * 0.93, 0.26), 0.20, uLights[2]);
	col += light(q, vec2(aspect * 0.12, 0.12), 0.34, uLights[3]);
	return col;
}

void main() {
	vec2 texel = 1.0 / uSim;
	float h = texture(uX, vUv).r;
	float hl = texture(uX, vUv - vec2(texel.x, 0.0)).r;
	float hr = texture(uX, vUv + vec2(texel.x, 0.0)).r;
	float hd = texture(uX, vUv - vec2(0.0, texel.y)).r;
	float hu = texture(uX, vUv + vec2(0.0, texel.y)).r;
	vec2 slope = vec2(hr - hl, hu - hd) * 0.5;
	float t = texture(uT, vUv).r;
	float sigma = sin(1.5707963 * t);

	vec3 n = normalize(vec3(-slope * 2.4, 1.0));
	float body = smoothstep(0.03, 0.16, h);
	vec2 q = vUv * vec2(uRes.x / uRes.y, 1.0);

	// A drop is a small lens: it shows the lights behind it, magnified and bent.
	vec3 behind = scene(q);
	vec3 lens = scene(q - n.xy * (0.10 + 0.14 * clamp(h, 0.0, 1.0)));
	vec3 col = mix(behind, lens * uLens + 0.01, body);

	// Its edge turns steep and goes dark; its top catches the light.
	float steep = 1.0 - n.z;
	col *= 1.0 - uRim * smoothstep(0.12, 0.55, steep) * body;
	vec3 toLight = normalize(vec3(-0.5, 0.62, 0.62));
	vec3 halfway = normalize(toLight + vec3(0.0, 0.0, 1.0));
	float spec = pow(max(dot(n, halfway), 0.0), 64.0);
	col += uSpec * spec * (0.12 + 0.88 * body);
	vec3 bounce = normalize(vec3(0.45, -0.55, 0.7));
	col += uBounce * pow(max(dot(n, normalize(bounce + vec3(0.0, 0.0, 1.0))), 0.0), 18.0) * body * 0.25;

	// Fog: microscopic drops scatter the light, whiter and flatter the higher the noise.
	vec3 fog = behind * uFogKeep + uFog + spec * 0.05;
	col = mix(col, fog, clamp(pow(sigma, 1.3) * 0.92, 0.0, 1.0));

	vec2 v = vUv - 0.5;
	col *= 1.0 - uVignette * dot(v, v);
	outColor = vec4(pow(clamp(col, 0.0, 1.0), vec3(1.0 / uGamma)), 1.0);
}`;

/** A seeded random number generator (mulberry32). */
function random(seed: number): () => number {
	let a = seed >>> 0;
	return () => {
		a = (a + 0x6d2b79f5) >>> 0;
		let t = a;
		t = Math.imul(t ^ (t >>> 15), t | 1);
		t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
}

interface Layout {
	/** Center of the wordmark and its width, as fractions of the field. */
	cx: number;
	cy: number;
	width: number;
	/** A box, in field fractions, where drops stay sparse so the copy reads. */
	quiet: [number, number, number, number];
}

function layoutFor(width: number, height: number): Layout {
	if (width < height) return { cx: 0.5, cy: 0.66, width: 0.86, quiet: [0, 0, 1, 0.5] };
	return { cx: 0.66, cy: 0.68, width: 0.5, quiet: [0, 0, 0.52, 0.56] };
}

/** Distance from each inside pixel to the nearest outside pixel (a 3-4 chamfer transform). */
function insideDistance(mask: Uint8Array, w: number, h: number): Float32Array {
	const big = 1e6;
	const d = new Float32Array(w * h);
	for (let i = 0; i < w * h; i++) d[i] = mask[i] ? big : 0;
	const at = (x: number, y: number) => (x < 0 || y < 0 || x >= w || y >= h ? 0 : d[y * w + x]);
	for (let y = 0; y < h; y++) {
		for (let x = 0; x < w; x++) {
			const i = y * w + x;
			if (!d[i]) continue;
			d[i] = Math.min(d[i], at(x - 1, y) + 3, at(x, y - 1) + 3, at(x - 1, y - 1) + 4, at(x + 1, y - 1) + 4);
		}
	}
	for (let y = h - 1; y >= 0; y--) {
		for (let x = w - 1; x >= 0; x--) {
			const i = y * w + x;
			if (!d[i]) continue;
			d[i] = Math.min(d[i], at(x + 1, y) + 3, at(x, y + 1) + 3, at(x + 1, y + 1) + 4, at(x - 1, y + 1) + 4);
		}
	}
	for (let i = 0; i < w * h; i++) d[i] /= 3;
	return d;
}

interface Drop {
	x: number;
	y: number;
	r: number;
	height: number;
}

/**
 * The clean state x0: a height field of water drops, packed inside the glyphs of
 * `word` and scattered thinly over the rest of the pane. Rows run bottom to top,
 * as WebGL reads them.
 */
function dropField(w: number, h: number, word: string, family: string, seed: number): Float32Array {
	const rand = random(seed);
	const layout = layoutFor(w, h);

	const canvas = document.createElement('canvas');
	canvas.width = w;
	canvas.height = h;
	const ctx = canvas.getContext('2d', { willReadFrequently: true })!;
	ctx.font = `600 100px ${family}`;
	const probe = ctx.measureText(word);
	const size = (100 * layout.width * w) / probe.width;
	ctx.font = `600 ${size}px ${family}`;
	const metrics = ctx.measureText(word);
	const top = metrics.actualBoundingBoxAscent;
	const bottom = metrics.actualBoundingBoxDescent;
	ctx.fillStyle = '#fff';
	ctx.textAlign = 'center';
	ctx.textBaseline = 'alphabetic';
	// Canvas rows run top to bottom; the field's run bottom to top.
	ctx.fillText(word, layout.cx * w, (1 - layout.cy) * h + (top - bottom) / 2);
	const pixels = ctx.getImageData(0, 0, w, h).data;
	const mask = new Uint8Array(w * h);
	for (let y = 0; y < h; y++) {
		for (let x = 0; x < w; x++) mask[(h - 1 - y) * w + x] = pixels[(y * w + x) * 4 + 3] > 127 ? 1 : 0;
	}
	const inside = insideDistance(mask, w, h);

	// A grid of accepted drops, so each new drop checks only its neighbours.
	const cell = 8;
	const gw = Math.ceil(w / cell);
	const gh = Math.ceil(h / cell);
	const grid: Drop[][] = Array.from({ length: gw * gh }, () => []);
	const drops: Drop[] = [];
	const fits = (x: number, y: number, r: number, gap: number) => {
		const reach = Math.ceil((r + 24 + gap) / cell);
		const gx = Math.floor(x / cell);
		const gy = Math.floor(y / cell);
		for (let j = Math.max(0, gy - reach); j <= Math.min(gh - 1, gy + reach); j++) {
			for (let i = Math.max(0, gx - reach); i <= Math.min(gw - 1, gx + reach); i++) {
				for (const other of grid[j * gw + i]) {
					const dx = other.x - x;
					const dy = other.y - y;
					const min = other.r + r + gap;
					if (dx * dx + dy * dy < min * min) return false;
				}
			}
		}
		return true;
	};
	const add = (drop: Drop) => {
		drops.push(drop);
		grid[Math.min(gh - 1, Math.floor(drop.y / cell)) * gw + Math.min(gw - 1, Math.floor(drop.x / cell))].push(drop);
	};

	// Inside the letters: beads as wide as the stroke allows, then smaller ones in the gaps.
	const strokeMax = Math.max(2, Math.min(24, w / 40));
	const insidePixels: number[] = [];
	for (let i = 0; i < w * h; i++) if (inside[i] >= 1.2) insidePixels.push(i);
	for (const [share, low, high] of [
		[0.35, 0.72, 0.98],
		[0.45, 0.4, 0.72],
		[0.6, 0.2, 0.4],
	] as const) {
		for (let k = 0; k < insidePixels.length * share; k++) {
			const i = insidePixels[Math.floor(rand() * insidePixels.length)];
			const x = (i % w) + rand() - 0.5;
			const y = Math.floor(i / w) + rand() - 0.5;
			const room = Math.min(inside[i], strokeMax);
			const r = room * (low + (high - low) * rand());
			if (r < 0.9 || !fits(x, y, r, 0.6)) continue;
			add({ x, y, r, height: Math.min(1, 0.4 + r / (strokeMax * 1.1)) });
		}
	}

	// Across the pane: a fine mist of small drops, a few medium ones, fewer behind the copy.
	const [qx0, qy0, qx1, qy1] = layout.quiet;
	const scatter = (w * h) / 40;
	for (let k = 0; k < scatter; k++) {
		const x = rand() * w;
		const y = rand() * h;
		if (inside[Math.floor(y) * w + Math.floor(x)] > 0) continue;
		const quiet = x / w >= qx0 && x / w <= qx1 && y / h >= qy0 && y / h <= qy1;
		if (quiet && rand() < 0.75) continue;
		const r = 0.55 + Math.pow(rand(), 5) * strokeMax * 0.3;
		if (!fits(x, y, r, 1.2)) continue;
		add({ x, y, r, height: Math.min(0.55, 0.16 + r / (strokeMax * 1.8)) });
	}

	// A thin film of water joins the beads inside each letter, so the word reads as one wet stroke.
	// Channel 0 is the height of water, channel 1 marks the letters.
	const field = new Float32Array(w * h * 2);
	for (let i = 0; i < w * h; i++) {
		if (inside[i] > 0) {
			field[2 * i] = 0.13 * Math.min(1, inside[i] / 2);
			field[2 * i + 1] = 1;
		}
	}
	for (const drop of drops) {
		const x0 = Math.max(0, Math.floor(drop.x - drop.r - 1));
		const x1 = Math.min(w - 1, Math.ceil(drop.x + drop.r + 1));
		const y0 = Math.max(0, Math.floor(drop.y - drop.r - 1));
		const y1 = Math.min(h - 1, Math.ceil(drop.y + drop.r + 1));
		for (let y = y0; y <= y1; y++) {
			for (let x = x0; x <= x1; x++) {
				const dx = x + 0.5 - drop.x;
				const dy = y + 0.5 - drop.y;
				const d = Math.sqrt(dx * dx + dy * dy) / drop.r;
				if (d >= 1.08) continue;
				// A sessile drop: a spherical cap, its rim softened over one pixel.
				const cap = Math.sqrt(Math.max(0, 1 - d * d));
				const edge = Math.min(1, Math.max(0, (1.08 - d) * drop.r));
				const value = drop.height * Math.max(cap, 0.08) * edge;
				const i = 2 * (y * w + x);
				if (value > field[i]) field[i] = value;
			}
		}
	}
	return field;
}

type Color = [number, number, number];

interface Palette {
	/** The scene behind the glass: a base color, what it gains toward the top, and four soft lights. */
	base: Color;
	rise: Color;
	lights: [Color, Color, Color, Color];
	/** How much a drop brightens what it shows, and how dark its steep rim turns. */
	lens: number;
	rim: number;
	spec: Color;
	bounce: Color;
	/** Fog: its own color, and how much of the scene shows through it. */
	fog: Color;
	fogKeep: number;
	vignette: number;
	gamma: number;
}

const PALETTES: Record<Theme, Palette> = {
	dark: {
		base: [0.012, 0.022, 0.026],
		rise: [0.01, 0.022, 0.024],
		lights: [
			[0.05, 0.3, 0.28],
			[0.03, 0.17, 0.21],
			[0.26, 0.17, 0.07],
			[0.02, 0.08, 0.09],
		],
		lens: 1.25,
		rim: 0.62,
		spec: [0.8, 0.96, 1.0],
		bounce: [0.2, 0.55, 0.52],
		fog: [0.07, 0.088, 0.092],
		fogKeep: 0.55,
		vignette: 0.35,
		gamma: 1.08,
	},
	light: {
		base: [0.9, 0.94, 0.935],
		rise: [0.05, 0.04, 0.045],
		lights: [
			[-0.34, -0.1, -0.14],
			[-0.2, -0.1, -0.03],
			[0.02, -0.08, -0.2],
			[-0.12, -0.05, -0.06],
		],
		lens: 1.0,
		rim: 0.5,
		spec: [0.5, 0.55, 0.55],
		bounce: [0.04, 0.12, 0.11],
		fog: [0.64, 0.665, 0.665],
		fogKeep: 0.34,
		vignette: 0.08,
		gamma: 1.0,
	},
};

export type Theme = 'dark' | 'light';

export interface Condensation {
	setTheme(theme: Theme): void;
	stop(): void;
}

/**
 * Run the condensation on `canvas`, filling `hero`, with the drops spelling
 * `word` in the CSS font `family`, which must have loaded. Returns null when
 * the browser lacks WebGL2 with float render targets.
 */
export function startCondensation(hero: HTMLElement, canvas: HTMLCanvasElement, word: string, family: string, theme: Theme): Condensation | null {
	const context = createContext(canvas);
	if (!context) return null;
	const { gl } = context;
	const reduced = prefersReducedMotion();
	let palette = PALETTES[theme];

	const vao = gl.createVertexArray()!;
	const tUpdate = createProgram(gl, FULLSCREEN_VERTEX, T_UPDATE);
	const xPass = createProgram(gl, FULLSCREEN_VERTEX, X_PASS);
	const render = createProgram(gl, FULLSCREEN_VERTEX, RENDER);

	let simW = 0;
	let simH = 0;
	let x0: WebGLTexture | null = null;
	let fieldX: Target | null = null;
	let times: [Target, Target] | null = null;
	const seed = (Math.random() * 2 ** 31) >>> 0;

	const build = () => {
		fitCanvas(canvas, reduced ? 1.25 : 1.5);
		const w = Math.max(64, Math.min(1100, Math.round(canvas.clientWidth / SIM_SCALE)));
		const h = Math.max(64, Math.round((w * canvas.clientHeight) / Math.max(1, canvas.clientWidth)));
		if (w === simW && h === simH) return;
		simW = w;
		simH = h;
		if (x0) gl.deleteTexture(x0);
		if (fieldX) destroyTarget(gl, fieldX);
		if (times) times.forEach((target) => destroyTarget(gl, target));
		x0 = createTexture(gl, w, h, {
			internalFormat: gl.RG16F,
			format: gl.RG,
			type: gl.FLOAT,
			filter: gl.LINEAR,
			data: dropField(w, h, word, family, seed),
		});
		fieldX = createTarget(gl, w, h);
		const tw = Math.ceil(w / TIME_SCALE);
		const th = Math.ceil(h / TIME_SCALE);
		times = [createTarget(gl, tw, th), createTarget(gl, tw, th)];
		// The page opens in fog; after a resize, the pane resumes where the schedule is.
		const fog = reduced ? 0 : started ? Math.max(0, 1 - elapsed / DURATION) : 1;
		for (const target of times) {
			gl.bindFramebuffer(gl.FRAMEBUFFER, target.framebuffer);
			gl.clearColor(fog, 0, 0, 1);
			gl.clear(gl.COLOR_BUFFER_BIT);
		}
		gl.bindFramebuffer(gl.FRAMEBUFFER, null);
	};

	// The pointer's path since the last frame, in field coordinates (y up).
	let stroke: [number, number, number, number] | null = null;
	let strokeStrength = 0;
	let lastPointer: [number, number] | null = null;
	const onPointer = (event: PointerEvent) => {
		const rect = canvas.getBoundingClientRect();
		const p: [number, number] = [(event.clientX - rect.left) / rect.width, 1 - (event.clientY - rect.top) / rect.height];
		const from = lastPointer ?? p;
		const speed = Math.hypot((p[0] - from[0]) * rect.width, (p[1] - from[1]) * rect.height);
		lastPointer = p;
		stroke = stroke ? [stroke[0], stroke[1], p[0], p[1]] : [from[0], from[1], p[0], p[1]];
		strokeStrength = Math.max(strokeStrength, Math.min(0.85, 0.4 + speed / 60));
	};
	const onLeave = () => (lastPointer = null);
	if (!reduced) {
		hero.addEventListener('pointermove', onPointer, { passive: true });
		hero.addEventListener('pointerleave', onLeave);
	}

	let elapsed = 0;
	let started = false;
	let noiseClock = 0;
	let resizeTimer = 0;
	const onResize = () => {
		clearTimeout(resizeTimer);
		resizeTimer = window.setTimeout(build, 150);
	};
	addEventListener('resize', onResize);

	build();

	const frame = (seconds: number, dt: number) => {
		if (!times || !fieldX || !x0) return;
		fitCanvas(canvas, reduced ? 1.25 : 1.5);
		started = true;
		elapsed = reduced ? DURATION : elapsed + dt;
		noiseClock += dt;
		const heroRect = hero.getBoundingClientRect();
		const scrollFog = Math.min(1, Math.max(0, -heroRect.top / (0.9 * heroRect.height)));

		// 1. Every pixel's clock steps back; the pointer and scrolling push it forward.
		const [prev, next] = times;
		gl.useProgram(tUpdate.program);
		bindTexture(gl, 0, prev.texture, tUpdate.uniform('uPrev'));
		bindTexture(gl, 1, x0, tUpdate.uniform('uX0'));
		gl.uniform1f(tUpdate.uniform('uDecay'), reduced ? 1 : dt / DURATION);
		gl.uniform1f(tUpdate.uniform('uFloor'), scrollFog);
		const s = stroke ?? [-9, -9, -9, -9];
		gl.uniform4f(tUpdate.uniform('uStroke'), s[0], s[1], s[2], s[3]);
		gl.uniform1f(tUpdate.uniform('uRadius'), 0.075);
		gl.uniform1f(tUpdate.uniform('uStrength'), stroke ? strokeStrength : 0);
		gl.uniform1f(tUpdate.uniform('uAspect'), canvas.clientWidth / Math.max(1, canvas.clientHeight));
		drawFullscreen(gl, vao, next);
		times = [next, prev];
		stroke = null;
		strokeStrength = 0;

		// 2. x_t = alpha(t) x0 + sigma(t) eps, with eps drifting from one noise field to the next.
		const period = 1.3;
		const k = Math.floor(noiseClock / period);
		gl.useProgram(xPass.program);
		bindTexture(gl, 0, x0, xPass.uniform('uX0'));
		bindTexture(gl, 1, next.texture, xPass.uniform('uT'));
		gl.uniform1ui(xPass.uniform('uSeedA'), (seed + k) >>> 0);
		gl.uniform1ui(xPass.uniform('uSeedB'), (seed + k + 1) >>> 0);
		gl.uniform1f(xPass.uniform('uMix'), ((noiseClock / period - k) * Math.PI) / 2);
		drawFullscreen(gl, vao, fieldX);

		// 3. Draw the pane.
		gl.useProgram(render.program);
		bindTexture(gl, 0, fieldX.texture, render.uniform('uX'));
		bindTexture(gl, 1, next.texture, render.uniform('uT'));
		gl.uniform2f(render.uniform('uSim'), simW, simH);
		gl.uniform2f(render.uniform('uRes'), canvas.width, canvas.height);
		gl.uniform1f(render.uniform('uTime'), seconds);
		gl.uniform3fv(render.uniform('uBase'), palette.base);
		gl.uniform3fv(render.uniform('uRise'), palette.rise);
		gl.uniform3fv(render.uniform('uLights'), palette.lights.flat());
		gl.uniform1f(render.uniform('uLens'), palette.lens);
		gl.uniform1f(render.uniform('uRim'), palette.rim);
		gl.uniform3fv(render.uniform('uSpec'), palette.spec);
		gl.uniform3fv(render.uniform('uBounce'), palette.bounce);
		gl.uniform3fv(render.uniform('uFog'), palette.fog);
		gl.uniform1f(render.uniform('uFogKeep'), palette.fogKeep);
		gl.uniform1f(render.uniform('uVignette'), palette.vignette);
		gl.uniform1f(render.uniform('uGamma'), palette.gamma);
		drawFullscreen(gl, vao, null);
	};

	if (reduced) {
		frame(0, 0);
		return {
			setTheme(next) {
				palette = PALETTES[next];
				frame(0, 0);
			},
			stop() {
				removeEventListener('resize', onResize);
			},
		};
	}
	const stopLoop = runLoop(canvas, frame);
	return {
		setTheme(next) {
			palette = PALETTES[next];
		},
		stop() {
			stopLoop();
			removeEventListener('resize', onResize);
			hero.removeEventListener('pointermove', onPointer);
			hero.removeEventListener('pointerleave', onLeave);
		},
	};
}
