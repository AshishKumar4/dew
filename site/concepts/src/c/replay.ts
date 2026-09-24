// Concept C: the page replays real sampling trajectories.
//
// Each figure is one prompt's trajectory from a text-to-image model sampled
// with Dew: at every step, the image the model predicts (x0) and, on request,
// the noisy latent decoded to pixels (x_t). The frames arrive as atlases; a
// WebGL2 shader shows the frame at a fractional step, blending neighbours, so
// scrolling or time can run the sampler forward and back.

import { FULLSCREEN_VERTEX, createContext, createProgram, fitCanvas } from '../lib/gl';

export interface TrajectoryMeta {
	prompt: string;
	seed: number | null;
	sampler: string | null;
	steps: number | null;
	cfg: number | null;
	frames: number;
	kept_steps: number[];
	sigmas: number[] | null;
	columns: number;
	size: number;
	placeholder: boolean;
}

const FRAGMENT = `#version 300 es
precision highp float;
in vec2 vUv;
uniform sampler2D uX0;
uniform sampler2D uXt;
uniform vec2 uGrid;      // columns, rows of the atlases
uniform float uStep;     // fractional frame index
uniform float uFrames;
uniform float uSize;     // pixels per frame
uniform float uNoisy;    // 0 shows the predicted x0, 1 the noisy x_t
uniform float uHasXt;
out vec4 outColor;

vec2 cell(float frame, vec2 uv) {
	float column = mod(frame, uGrid.x);
	float row = floor(frame / uGrid.x);
	// Half a texel of margin keeps linear filtering inside the frame.
	vec2 inner = mix(vec2(0.5 / uSize), vec2(1.0 - 0.5 / uSize), uv);
	return (vec2(column, row) + inner) / uGrid;
}

void main() {
	vec2 uv = vec2(vUv.x, 1.0 - vUv.y);
	float a = floor(uStep);
	float b = min(a + 1.0, uFrames - 1.0);
	float f = uStep - a;
	vec3 x0 = mix(texture(uX0, cell(a, uv)).rgb, texture(uX0, cell(b, uv)).rgb, f);
	vec3 color = x0;
	if (uHasXt > 0.5) {
		vec3 xt = mix(texture(uXt, cell(a, uv)).rgb, texture(uXt, cell(b, uv)).rgb, f);
		color = mix(x0, xt, uNoisy);
	}
	outColor = vec4(color, 1.0);
}`;

function loadImage(url: string): Promise<HTMLImageElement> {
	return new Promise((resolve, reject) => {
		const image = new Image();
		image.decoding = 'async';
		image.onload = () => resolve(image);
		image.onerror = () => reject(new Error(`could not load ${url}`));
		image.src = url;
	});
}

export class Replay {
	readonly meta: TrajectoryMeta;
	private readonly gl: WebGL2RenderingContext;
	private readonly canvas: HTMLCanvasElement;
	private readonly program;
	private readonly vao: WebGLVertexArrayObject;
	private readonly x0: WebGLTexture;
	private xt: WebGLTexture | null = null;
	private xtLoading: Promise<void> | null = null;
	private readonly base: string;
	step = 0;
	noisy = 0;

	private constructor(canvas: HTMLCanvasElement, gl: WebGL2RenderingContext, meta: TrajectoryMeta, x0: HTMLImageElement, base: string) {
		this.canvas = canvas;
		this.gl = gl;
		this.meta = meta;
		this.base = base;
		this.program = createProgram(gl, FULLSCREEN_VERTEX, FRAGMENT);
		this.vao = gl.createVertexArray()!;
		this.x0 = this.upload(x0);
	}

	/** Load the trajectory in `base` (meta.json, x0.webp) onto `canvas`, or null without WebGL2. */
	static async load(canvas: HTMLCanvasElement, base: string): Promise<Replay | null> {
		const context = createContext(canvas, false);
		if (!context) return null;
		const [meta, x0] = await Promise.all([
			fetch(`${base}/meta.json`).then((response) => response.json() as Promise<TrajectoryMeta>),
			loadImage(`${base}/x0.webp`),
		]);
		return new Replay(canvas, context.gl, meta, x0, base);
	}

	private upload(image: HTMLImageElement): WebGLTexture {
		const { gl } = this;
		const texture = gl.createTexture()!;
		gl.bindTexture(gl.TEXTURE_2D, texture);
		gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
		gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, image);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
		return texture;
	}

	/** Fetch the noisy frames, once, the first time someone asks to see them. */
	loadNoisy(): Promise<void> {
		this.xtLoading ??= loadImage(`${this.base}/xt.webp`).then((image) => {
			this.xt = this.upload(image);
		});
		return this.xtLoading;
	}

	get hasNoisy(): boolean {
		return this.xt !== null;
	}

	/** The sampler step and sigma shown at the current fractional frame. */
	readout(): { step: number; steps: number; sigma: number | null } {
		const frame = Math.round(this.step);
		const { kept_steps: kept, sigmas } = this.meta;
		return { step: kept[frame], steps: kept[kept.length - 1], sigma: sigmas ? sigmas[frame] : null };
	}

	draw(): void {
		const { gl } = this;
		fitCanvas(this.canvas, 2);
		gl.bindFramebuffer(gl.FRAMEBUFFER, null);
		gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
		gl.useProgram(this.program.program);
		gl.activeTexture(gl.TEXTURE0);
		gl.bindTexture(gl.TEXTURE_2D, this.x0);
		gl.uniform1i(this.program.uniform('uX0'), 0);
		gl.activeTexture(gl.TEXTURE1);
		gl.bindTexture(gl.TEXTURE_2D, this.xt ?? this.x0);
		gl.uniform1i(this.program.uniform('uXt'), 1);
		const rows = Math.ceil(this.meta.frames / this.meta.columns);
		gl.uniform2f(this.program.uniform('uGrid'), this.meta.columns, rows);
		gl.uniform1f(this.program.uniform('uStep'), Math.max(0, Math.min(this.meta.frames - 1, this.step)));
		gl.uniform1f(this.program.uniform('uFrames'), this.meta.frames);
		gl.uniform1f(this.program.uniform('uSize'), this.meta.size);
		gl.uniform1f(this.program.uniform('uNoisy'), this.noisy);
		gl.uniform1f(this.program.uniform('uHasXt'), this.xt ? 1 : 0);
		gl.bindVertexArray(this.vao);
		gl.drawArrays(gl.TRIANGLES, 0, 3);
	}
}
