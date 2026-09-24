// The landing page's replay section: a text-to-image model's sampling
// trajectory, played back as the section scrolls past. The frames are the
// model's predicted image at kept sampler steps, packed into one atlas by
// scripts/replay-atlas.py; a WebGL2 shader shows the frame at a fractional
// step, blending neighbours, so scrolling runs the sampler forward and back.

import { FULLSCREEN_VERTEX, createContext, createProgram, fitCanvas, type Program } from './gl';

export interface ReplayMeta {
	prompt: string;
	frames: number;
	columns: number;
	size: number;
}

const FRAGMENT = `#version 300 es
precision highp float;
in vec2 vUv;
uniform sampler2D uAtlas;
uniform vec2 uGrid;      // columns, rows of the atlas
uniform float uStep;     // fractional frame index
uniform float uFrames;
uniform float uSize;     // pixels per frame
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
	vec3 color = mix(texture(uAtlas, cell(a, uv)).rgb, texture(uAtlas, cell(b, uv)).rgb, uStep - a);
	outColor = vec4(color, 1.0);
}`;

function loadImage(url: string): Promise<HTMLImageElement> {
	const { promise, resolve, reject } = Promise.withResolvers<HTMLImageElement>();
	const image = new Image();
	image.decoding = 'async';
	image.onload = () => resolve(image);
	image.onerror = () => reject(new Error(`could not load ${url}`));
	image.src = url;
	return promise;
}

export class Replay {
	readonly meta: ReplayMeta;
	/** The frame shown: 0 is the first sampler step, meta.frames - 1 the sample. */
	step = 0;
	private readonly gl: WebGL2RenderingContext;
	private readonly canvas: HTMLCanvasElement;
	private readonly program: Program;
	private readonly vao: WebGLVertexArrayObject;
	private readonly atlas: WebGLTexture;

	private constructor(canvas: HTMLCanvasElement, gl: WebGL2RenderingContext, meta: ReplayMeta, atlas: HTMLImageElement) {
		this.canvas = canvas;
		this.gl = gl;
		this.meta = meta;
		this.program = createProgram(gl, FULLSCREEN_VERTEX, FRAGMENT);
		this.vao = gl.createVertexArray()!;
		this.atlas = gl.createTexture()!;
		gl.bindTexture(gl.TEXTURE_2D, this.atlas);
		gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, atlas);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
	}

	/** Load the trajectory in `base` (meta.json, x0.webp) onto `canvas`, or null without a hardware WebGL2. */
	static async load(canvas: HTMLCanvasElement, base: string): Promise<Replay | null> {
		const context = createContext(canvas, false);
		if (!context) return null;
		const [meta, atlas] = await Promise.all([
			fetch(`${base}/meta.json`).then((response) => response.json() as Promise<ReplayMeta>),
			loadImage(`${base}/x0.webp`),
		]);
		return new Replay(canvas, context.gl, meta, atlas);
	}

	draw(): void {
		const { gl, meta } = this;
		fitCanvas(this.canvas, 2);
		gl.bindFramebuffer(gl.FRAMEBUFFER, null);
		gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
		gl.useProgram(this.program.program);
		gl.activeTexture(gl.TEXTURE0);
		gl.bindTexture(gl.TEXTURE_2D, this.atlas);
		gl.uniform1i(this.program.uniform('uAtlas'), 0);
		gl.uniform2f(this.program.uniform('uGrid'), meta.columns, Math.ceil(meta.frames / meta.columns));
		gl.uniform1f(this.program.uniform('uStep'), Math.max(0, Math.min(meta.frames - 1, this.step)));
		gl.uniform1f(this.program.uniform('uFrames'), meta.frames);
		gl.uniform1f(this.program.uniform('uSize'), meta.size);
		gl.bindVertexArray(this.vao);
		gl.drawArrays(gl.TRIANGLES, 0, 3);
	}
}
