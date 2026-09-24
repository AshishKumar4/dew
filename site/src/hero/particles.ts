// One of the landing page's two heroes: a model that runs on the visitor's GPU.
//
// A small velocity network, trained with Dew by rectified flow on points drawn
// from the letters of "dew", moves a cloud of Gaussian noise into the word.
// Each particle is one sample. Every frame the network runs on the GPU for all
// particles at once, layer by layer in fragment shaders over float textures,
// and each particle takes one Euler step from t = 1 toward t = 0. The pointer
// puts noise back into the particles it passes, and the network brings them home.

import { FULLSCREEN_VERTEX, createContext, createProgram, fitCanvas, prefersReducedMotion, runLoop, type Program } from './gl';

export interface Manifest {
	layers: { name: string; in: number; out: number }[];
	position_freqs: number;
	time_freqs: number;
	parameters: number;
	sample_steps: number;
	train_steps: number;
	train_seconds: number;
	samples_inside: number;
	dew_commit: string | null;
	device: string;
}

const DURATION = 2.4; // seconds from noise to the word
const GRID_POWER = 1.5; // t = s^1.5: finer steps near the data, as the training script samples
const ROW = 128; // particles per row of the state textures

interface Texture2D {
	texture: WebGLTexture;
	framebuffer: WebGLFramebuffer;
	width: number;
	height: number;
}

function floatTarget(gl: WebGL2RenderingContext, width: number, height: number): Texture2D {
	const texture = gl.createTexture()!;
	gl.bindTexture(gl.TEXTURE_2D, texture);
	gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, width, height, 0, gl.RGBA, gl.FLOAT, null);
	gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
	gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
	gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
	gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
	const framebuffer = gl.createFramebuffer()!;
	gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
	gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
	if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) throw new Error('float framebuffer incomplete');
	gl.bindFramebuffer(gl.FRAMEBUFFER, null);
	return { texture, framebuffer, width, height };
}

/**
 * One layer's weights as a texture of width ceil(out / 4) and height in + 1:
 * texel (o, i) holds kernel[i][4o .. 4o + 3], and the last row holds the bias.
 */
function layerTexture(gl: WebGL2RenderingContext, inputs: number, outputs: number, kernel: Float32Array, bias: Float32Array): WebGLTexture {
	const groups = Math.ceil(outputs / 4);
	const data = new Float32Array(groups * 4 * (inputs + 1));
	for (let i = 0; i <= inputs; i++) {
		for (let j = 0; j < outputs; j++) {
			data[(i * groups + (j >> 2)) * 4 + (j & 3)] = i < inputs ? kernel[i * outputs + j] : bias[j];
		}
	}
	const texture = gl.createTexture()!;
	gl.bindTexture(gl.TEXTURE_2D, texture);
	gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, groups, inputs + 1, 0, gl.RGBA, gl.FLOAT, data);
	gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
	gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
	return texture;
}

const HASH = `
uint hash_u(uint x) {
	x ^= x >> 16; x *= 0x7feb352du;
	x ^= x >> 15; x *= 0x846ca68bu;
	x ^= x >> 16;
	return x;
}
vec2 gauss2(uint id, uint stream) {
	uint h1 = hash_u(id * 2u + hash_u(stream));
	uint h2 = hash_u(h1 ^ 0x9e3779b9u);
	float u1 = (float(h1 >> 8) + 0.5) / 16777216.0;
	float u2 = (float(h2 >> 8) + 0.5) / 16777216.0;
	return sqrt(-2.0 * log(u1)) * vec2(cos(6.28318530718 * u2), sin(6.28318530718 * u2));
}`;

/** Features of each particle: x, y, then sin and cos of 2^k pi x per coordinate, then sin and cos of 2^q pi t / 2. */
function featureShader(inputs: number, pf: number, tf: number): string {
	const groups = Math.ceil(inputs / 4);
	return `#version 300 es
precision highp float;
precision highp int;
uniform highp sampler2D uState;
out vec4 outValue;
const float PI = 3.14159265359;
float feature(int n, vec2 x, float t) {
	if (n == 0) return x.x;
	if (n == 1) return x.y;
	n -= 2;
	if (n < ${4 * pf}) {
		int d = n / ${2 * pf};
		int r = n - d * ${2 * pf};
		float xd = d == 0 ? x.x : x.y;
		return r < ${pf} ? sin(xd * exp2(float(r)) * PI) : cos(xd * exp2(float(r - ${pf})) * PI);
	}
	n -= ${4 * pf};
	if (n < ${2 * tf}) return n < ${tf} ? sin(t * exp2(float(n)) * PI * 0.5) : cos(t * exp2(float(n - ${tf})) * PI * 0.5);
	return 0.0;
}
void main() {
	ivec2 p = ivec2(gl_FragCoord.xy);
	int group = p.x % ${groups};
	vec4 state = texelFetch(uState, ivec2(p.x / ${groups}, p.y), 0);
	float t = pow(state.z, ${GRID_POWER.toFixed(2)});
	outValue = vec4(feature(group * 4, state.xy, t), feature(group * 4 + 1, state.xy, t),
	                feature(group * 4 + 2, state.xy, t), feature(group * 4 + 3, state.xy, t));
}`;
}

/** out[p][4o .. 4o+3] = bias + sum_i in[p][i] * W[i][4o .. 4o+3], then SiLU unless it is the last layer. */
function denseShader(inputs: number, outputs: number, activate: boolean): string {
	const inGroups = Math.ceil(inputs / 4);
	const outGroups = Math.ceil(outputs / 4);
	const whole = Math.floor(inputs / 4);
	const rest = inputs - whole * 4;
	return `#version 300 es
precision highp float;
precision highp int;
uniform highp sampler2D uIn;
uniform highp sampler2D uW;
out vec4 outValue;
void main() {
	ivec2 p = ivec2(gl_FragCoord.xy);
	int particle = p.x / ${outGroups};
	int o = p.x - particle * ${outGroups};
	int base = particle * ${inGroups};
	vec4 acc = texelFetch(uW, ivec2(o, ${inputs}), 0);
	for (int g = 0; g < ${whole}; g++) {
		vec4 a = texelFetch(uIn, ivec2(base + g, p.y), 0);
		acc += a.x * texelFetch(uW, ivec2(o, 4 * g), 0);
		acc += a.y * texelFetch(uW, ivec2(o, 4 * g + 1), 0);
		acc += a.z * texelFetch(uW, ivec2(o, 4 * g + 2), 0);
		acc += a.w * texelFetch(uW, ivec2(o, 4 * g + 3), 0);
	}
	${
		rest > 0
			? `vec4 last = texelFetch(uIn, ivec2(base + ${whole}, p.y), 0);
	${Array.from({ length: rest }, (_, k) => `acc += last[${k}] * texelFetch(uW, ivec2(o, ${whole * 4 + k}), 0);`).join('\n\t')}`
			: ''
	}
	outValue = ${activate ? 'acc / (1.0 + exp(-acc))' : 'acc'};
}`;
}

/** One Euler step of the flow per particle, after the pointer's noise. */
const UPDATE_SHADER = `#version 300 es
precision highp float;
precision highp int;
uniform highp sampler2D uState;
uniform highp sampler2D uVelocity;
uniform float uDs;
uniform vec4 uPointer;   // model-space x, y, radius, strength
uniform uint uGeneration;
out vec4 outValue;
${HASH}
void main() {
	ivec2 p = ivec2(gl_FragCoord.xy);
	vec4 state = texelFetch(uState, p, 0);
	vec2 x = state.xy;
	float s = state.z;
	if (s > 0.0) {
		float next = max(s - uDs, 0.0);
		float t = pow(s, ${GRID_POWER.toFixed(2)});
		float tn = pow(next, ${GRID_POWER.toFixed(2)});
		x += (tn - t) * texelFetch(uVelocity, p, 0).xy;
		s = next;
	}
	// The pointer puts noise back: a particle it passes jumps to a later time, x = (1 - t) x + t eps.
	if (uPointer.z > 0.0) {
		float d = length(x - uPointer.xy);
		float target = uPointer.w * exp(-(d * d) / (uPointer.z * uPointer.z));
		if (target > s + 0.02) {
			float t = pow(target, ${GRID_POWER.toFixed(2)});
			uint id = uint(p.y * ${ROW} + p.x);
			x = (1.0 - t) * x + t * gauss2(id, uGeneration);
			s = target;
		}
	}
	outValue = vec4(x, s, 0.0);
}`;

const DRAW_VERTEX = `#version 300 es
precision highp float;
precision highp int;
uniform highp sampler2D uState;
uniform vec4 uView;   // scale x, scale y, offset x, offset y: model space to clip space
uniform float uSize;
out float vS;
void main() {
	vec4 state = texelFetch(uState, ivec2(gl_VertexID % ${ROW}, gl_VertexID / ${ROW}), 0);
	vS = state.z;
	gl_Position = vec4(state.xy * uView.xy + uView.zw, 0.0, 1.0);
	gl_PointSize = uSize * mix(1.0, 0.72, state.z);
}`;

const DRAW_FRAGMENT = `#version 300 es
precision highp float;
in float vS;
uniform vec3 uSettled;
uniform vec3 uNoisy;
uniform vec2 uAlpha;     // opacity of a noisy and of a settled particle
uniform float uHighlight;
out vec4 outColor;
void main() {
	vec2 c = gl_PointCoord * 2.0 - 1.0;
	float r2 = dot(c, c);
	if (r2 > 1.0) discard;
	// A droplet: a soft body and a highlight up and to the left.
	float body = smoothstep(1.0, 0.45, r2);
	vec2 h = c - vec2(-0.32, -0.32);
	float highlight = exp(-dot(h, h) * 10.0);
	float wet = smoothstep(0.8, 0.0, vS);
	vec3 color = mix(uNoisy, uSettled, wet);
	float alpha = body * mix(uAlpha.x, uAlpha.y, wet);
	outColor = vec4(color * alpha + vec3(0.85, 1.0, 1.0) * highlight * uHighlight * wet * body, alpha);
}`;

export type Theme = 'dark' | 'light';

type Color = [number, number, number];

interface Palette {
	/** The page's background, which the canvas clears to. */
	background: Color;
	settled: Color;
	noisy: Color;
	alpha: [number, number];
	highlight: number;
}

const PALETTES: Record<Theme, Palette> = {
	// #0a1113 and #fbfcfb, Starlight's --sl-color-black in each theme (src/styles/theme.css).
	dark: { background: [0.039, 0.067, 0.075], settled: [0.46, 0.9, 0.84], noisy: [0.26, 0.52, 0.7], alpha: [0.5, 0.72], highlight: 0.45 },
	light: { background: [0.984, 0.988, 0.984], settled: [0.03, 0.47, 0.42], noisy: [0.3, 0.5, 0.64], alpha: [0.55, 0.9], highlight: 0.3 },
};

export interface ParticleField {
	setTheme(theme: Theme): void;
	stop(): void;
	resample(): void;
}

function placementFor(width: number, height: number): { cx: number; cy: number; width: number } {
	if (width < height) return { cx: 0.5, cy: 0.66, width: 0.84 };
	return { cx: 0.66, cy: 0.68, width: 0.5 };
}

/**
 * Run the particle model on `canvas`. Returns null when the browser lacks
 * WebGL2 with float render targets; the page then shows its still frame.
 */
export function startParticles(
	hero: HTMLElement,
	canvas: HTMLCanvasElement,
	manifest: Manifest,
	weights: Float32Array,
	theme: Theme,
): ParticleField | null {
	const context = createContext(canvas, true);
	if (!context) return null;
	const { gl } = context;
	let palette = PALETTES[theme];

	const reduced = prefersReducedMotion();
	const count = Math.min(window.innerWidth, window.innerHeight) < 600 ? 8192 : 16384;
	const rows = count / ROW;

	// The network's layers as textures, and one pass per layer.
	const [first] = manifest.layers;
	if (first.in !== 2 + 4 * manifest.position_freqs + 2 * manifest.time_freqs) throw new Error('the first layer does not take the features');
	let offset = 0;
	const layers = manifest.layers.map((layer, index) => {
		const kernel = weights.subarray(offset, offset + layer.in * layer.out);
		offset += layer.in * layer.out;
		const bias = weights.subarray(offset, offset + layer.out);
		offset += layer.out;
		const last = index === manifest.layers.length - 1;
		return {
			inGroups: Math.ceil(layer.in / 4),
			outGroups: Math.ceil(layer.out / 4),
			weights: layerTexture(gl, layer.in, layer.out, kernel, bias),
			program: createProgram(gl, FULLSCREEN_VERTEX, denseShader(layer.in, layer.out, !last)),
		};
	});
	if (offset !== weights.length) throw new Error(`weights: used ${offset} of ${weights.length} floats`);

	const features = createProgram(gl, FULLSCREEN_VERTEX, featureShader(first.in, manifest.position_freqs, manifest.time_freqs));
	const update = createProgram(gl, FULLSCREEN_VERTEX, UPDATE_SHADER);
	const draw = createProgram(gl, DRAW_VERTEX, DRAW_FRAGMENT);
	const vao = gl.createVertexArray()!;

	// Activations: the features, then each layer's output, all ROW particles wide.
	const activations = [
		floatTarget(gl, ROW * layers[0].inGroups, rows),
		...layers.map((layer) => floatTarget(gl, ROW * layer.outGroups, rows)),
	];
	const states = [floatTarget(gl, ROW, rows), floatTarget(gl, ROW, rows)];
	let current = 0;

	const pass = (program: Program, target: Texture2D, inputs: [string, WebGLTexture][]) => {
		gl.useProgram(program.program);
		inputs.forEach(([name, texture], unit) => {
			gl.activeTexture(gl.TEXTURE0 + unit);
			gl.bindTexture(gl.TEXTURE_2D, texture);
			gl.uniform1i(program.uniform(name), unit);
		});
		gl.bindFramebuffer(gl.FRAMEBUFFER, target.framebuffer);
		gl.viewport(0, 0, target.width, target.height);
		gl.bindVertexArray(vao);
		gl.drawArrays(gl.TRIANGLES, 0, 3);
	};

	let progress = 1;
	let generation = 1;
	let pointer: { x: number; y: number } | null = null;
	let activeUntil = 0; // a pointer's noise keeps the flow running until the particles settle again
	let clock = 0;

	const seedNoise = () => {
		const state = new Float32Array(count * 4);
		for (let i = 0; i < count; i++) {
			const u1 = Math.random() || 1e-7;
			const u2 = Math.random();
			const r = Math.sqrt(-2 * Math.log(u1));
			state[4 * i] = r * Math.cos(2 * Math.PI * u2);
			state[4 * i + 1] = r * Math.sin(2 * Math.PI * u2);
			state[4 * i + 2] = 1;
		}
		gl.bindTexture(gl.TEXTURE_2D, states[current].texture);
		gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, ROW, rows, gl.RGBA, gl.FLOAT, state);
		progress = 1;
	};

	const step = (ds: number) => {
		pass(features, activations[0], [['uState', states[current].texture]]);
		layers.forEach((layer, index) => {
			pass(layer.program, activations[index + 1], [
				['uIn', activations[index].texture],
				['uW', layer.weights],
			]);
		});
		gl.useProgram(update.program);
		gl.uniform1f(update.uniform('uDs'), ds);
		if (pointer) {
			gl.uniform4f(update.uniform('uPointer'), pointer.x, pointer.y, 0.16, 0.55);
			gl.uniform1ui(update.uniform('uGeneration'), generation++);
			pointer = null;
		} else {
			gl.uniform4f(update.uniform('uPointer'), 0, 0, 0, 0);
		}
		pass(update, states[1 - current], [
			['uState', states[current].texture],
			['uVelocity', activations[activations.length - 1].texture],
		]);
		current = 1 - current;
	};

	const view = () => {
		const { cx, cy, width } = placementFor(canvas.clientWidth, canvas.clientHeight);
		const scaleX = (width * 2) / 3.6; // the word spans x in [-1.8, 1.8] of model space
		const scaleY = (scaleX * canvas.clientWidth) / Math.max(1, canvas.clientHeight);
		return [scaleX, scaleY, cx * 2 - 1, cy * 2 - 1] as const;
	};

	const render = () => {
		fitCanvas(canvas, 2);
		gl.bindFramebuffer(gl.FRAMEBUFFER, null);
		gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
		gl.clearColor(...palette.background, 1);
		gl.clear(gl.COLOR_BUFFER_BIT);
		gl.enable(gl.BLEND);
		gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
		gl.useProgram(draw.program);
		gl.activeTexture(gl.TEXTURE0);
		gl.bindTexture(gl.TEXTURE_2D, states[current].texture);
		gl.uniform1i(draw.uniform('uState'), 0);
		gl.uniform3fv(draw.uniform('uSettled'), palette.settled);
		gl.uniform3fv(draw.uniform('uNoisy'), palette.noisy);
		gl.uniform2fv(draw.uniform('uAlpha'), palette.alpha);
		gl.uniform1f(draw.uniform('uHighlight'), palette.highlight);
		const [sx, sy, ox, oy] = view();
		gl.uniform4f(draw.uniform('uView'), sx, sy, ox, oy);
		// Points about as wide as the gaps between settled particles.
		const ratio = gl.drawingBufferWidth / Math.max(1, canvas.clientWidth);
		const wordArea = ((canvas.clientWidth * sx) / 2) * 3.6 * ((canvas.clientHeight * sy) / 2) * 1.6 * 0.33;
		const spacing = Math.sqrt(wordArea / count);
		gl.uniform1f(draw.uniform('uSize'), Math.max(2, Math.min(7, spacing * 1.6)) * ratio);
		gl.bindVertexArray(vao);
		gl.drawArrays(gl.POINTS, 0, count);
		gl.bindVertexArray(null);
		gl.disable(gl.BLEND);
	};

	const onPointer = (event: PointerEvent) => {
		const rect = canvas.getBoundingClientRect();
		const [sx, sy, ox, oy] = view();
		const clipX = ((event.clientX - rect.left) / rect.width) * 2 - 1;
		const clipY = 1 - ((event.clientY - rect.top) / rect.height) * 2;
		pointer = { x: (clipX - ox) / sx, y: (clipY - oy) / sy };
		activeUntil = clock + 0.55 * DURATION + 0.2;
	};

	seedNoise();

	const settleAll = () => {
		for (let i = 0; i < manifest.sample_steps; i++) step(1 / manifest.sample_steps);
		progress = 0;
		render();
	};

	if (reduced) {
		settleAll();
		return {
			setTheme(next) {
				palette = PALETTES[next];
				render();
			},
			stop() {},
			resample() {
				seedNoise();
				settleAll();
			},
		};
	}

	hero.addEventListener('pointermove', onPointer, { passive: true });
	const stopLoop = runLoop(canvas, (_seconds, dt) => {
		clock += dt;
		const ds = dt / DURATION;
		// Once every particle has settled and nothing stirs them, the network can rest.
		if (progress > 0 || pointer || clock < activeUntil) {
			progress = Math.max(0, progress - ds);
			step(ds);
		}
		render();
	});
	return {
		setTheme(next) {
			palette = PALETTES[next];
		},
		stop() {
			stopLoop();
			hero.removeEventListener('pointermove', onPointer);
		},
		resample() {
			seedNoise();
		},
	};
}
