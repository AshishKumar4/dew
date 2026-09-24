// A small WebGL2 toolkit for the landing page: programs, float
// render targets, a fullscreen triangle, and a frame loop that sleeps when the
// canvas is off screen or the tab is hidden.

export interface Context {
	gl: WebGL2RenderingContext;
	canvas: HTMLCanvasElement;
	/** Whether float textures can be render targets (EXT_color_buffer_float). */
	floatTargets: boolean;
}

export function prefersReducedMotion(): boolean {
	return matchMedia('(prefers-reduced-motion: reduce)').matches;
}

/** A WebGL2 context, or null when the browser has none or it lacks float render targets. */
export function createContext(canvas: HTMLCanvasElement, needFloatTargets = true): Context | null {
	const gl = canvas.getContext('webgl2', {
		alpha: false,
		antialias: false,
		depth: false,
		stencil: false,
		premultipliedAlpha: false,
		preserveDrawingBuffer: false,
		powerPreference: 'high-performance',
	});
	if (!gl) return null;
	const floatTargets = gl.getExtension('EXT_color_buffer_float') !== null;
	gl.getExtension('OES_texture_float_linear');
	if (needFloatTargets && !floatTargets) return null;
	return { gl, canvas, floatTargets };
}

export const FULLSCREEN_VERTEX = `#version 300 es
out vec2 vUv;
void main() {
	vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
	vUv = p;
	gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;

export interface Program {
	program: WebGLProgram;
	uniform(name: string): WebGLUniformLocation | null;
}

function compileShader(gl: WebGL2RenderingContext, type: number, source: string): WebGLShader {
	const shader = gl.createShader(type)!;
	gl.shaderSource(shader, source);
	gl.compileShader(shader);
	if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
		const log = gl.getShaderInfoLog(shader);
		const numbered = source
			.split('\n')
			.map((line, i) => `${String(i + 1).padStart(3)} ${line}`)
			.join('\n');
		throw new Error(`shader compile failed: ${log}\n${numbered}`);
	}
	return shader;
}

export function createProgram(
	gl: WebGL2RenderingContext,
	vertex: string,
	fragment: string,
	transformFeedback?: string[],
): Program {
	const program = gl.createProgram()!;
	gl.attachShader(program, compileShader(gl, gl.VERTEX_SHADER, vertex));
	gl.attachShader(program, compileShader(gl, gl.FRAGMENT_SHADER, fragment));
	if (transformFeedback) gl.transformFeedbackVaryings(program, transformFeedback, gl.INTERLEAVED_ATTRIBS);
	gl.linkProgram(program);
	if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
		throw new Error(`program link failed: ${gl.getProgramInfoLog(program)}`);
	}
	const cache = new Map<string, WebGLUniformLocation | null>();
	return {
		program,
		uniform(name) {
			if (!cache.has(name)) cache.set(name, gl.getUniformLocation(program, name));
			return cache.get(name)!;
		},
	};
}

export interface TextureOptions {
	internalFormat: number;
	format: number;
	type: number;
	filter?: number;
	wrap?: number;
	data?: ArrayBufferView | null;
}

export function createTexture(gl: WebGL2RenderingContext, width: number, height: number, options: TextureOptions): WebGLTexture {
	const texture = gl.createTexture()!;
	gl.bindTexture(gl.TEXTURE_2D, texture);
	gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
	gl.texImage2D(gl.TEXTURE_2D, 0, options.internalFormat, width, height, 0, options.format, options.type, options.data ?? null);
	const filter = options.filter ?? gl.LINEAR;
	const wrap = options.wrap ?? gl.CLAMP_TO_EDGE;
	gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
	gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
	gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, wrap);
	gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, wrap);
	return texture;
}

export interface Target {
	texture: WebGLTexture;
	framebuffer: WebGLFramebuffer;
	width: number;
	height: number;
}

/** A single-channel half-float texture with a framebuffer, the state of a simulated field. */
export function createTarget(gl: WebGL2RenderingContext, width: number, height: number, channels: 1 | 4 = 1): Target {
	const texture = createTexture(gl, width, height, {
		internalFormat: channels === 1 ? gl.R16F : gl.RGBA16F,
		format: channels === 1 ? gl.RED : gl.RGBA,
		type: gl.HALF_FLOAT,
	});
	const framebuffer = gl.createFramebuffer()!;
	gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
	gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
	const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
	gl.bindFramebuffer(gl.FRAMEBUFFER, null);
	if (status !== gl.FRAMEBUFFER_COMPLETE) throw new Error(`framebuffer incomplete: 0x${status.toString(16)}`);
	return { texture, framebuffer, width, height };
}

export function destroyTarget(gl: WebGL2RenderingContext, target: Target): void {
	gl.deleteTexture(target.texture);
	gl.deleteFramebuffer(target.framebuffer);
}

/** Draw the fullscreen triangle into `target`, or the canvas when it is null. */
export function drawFullscreen(gl: WebGL2RenderingContext, vao: WebGLVertexArrayObject, target: Target | null): void {
	gl.bindFramebuffer(gl.FRAMEBUFFER, target ? target.framebuffer : null);
	gl.viewport(0, 0, target ? target.width : gl.drawingBufferWidth, target ? target.height : gl.drawingBufferHeight);
	gl.bindVertexArray(vao);
	gl.drawArrays(gl.TRIANGLES, 0, 3);
}

export function bindTexture(gl: WebGL2RenderingContext, unit: number, texture: WebGLTexture, location: WebGLUniformLocation | null): void {
	gl.activeTexture(gl.TEXTURE0 + unit);
	gl.bindTexture(gl.TEXTURE_2D, texture);
	gl.uniform1i(location, unit);
}

/** Match the drawing buffer to the canvas's CSS size at a capped pixel ratio. Returns whether it changed. */
export function fitCanvas(canvas: HTMLCanvasElement, maxRatio: number): boolean {
	const ratio = Math.min(devicePixelRatio || 1, maxRatio);
	const width = Math.max(1, Math.round(canvas.clientWidth * ratio));
	const height = Math.max(1, Math.round(canvas.clientHeight * ratio));
	if (canvas.width === width && canvas.height === height) return false;
	canvas.width = width;
	canvas.height = height;
	return true;
}

/**
 * Call `frame(seconds, dt)` on every animation frame while the canvas is on
 * screen and the page is visible. Returns a function that stops the loop.
 */
export function runLoop(canvas: HTMLCanvasElement, frame: (seconds: number, dt: number) => void): () => void {
	let visible = true;
	let handle = 0;
	let last = -1;
	let stopped = false;
	const tick = (now: number) => {
		handle = 0;
		if (stopped) return;
		const seconds = now / 1000;
		const dt = last < 0 ? 1 / 60 : Math.min(seconds - last, 1 / 20);
		last = seconds;
		frame(seconds, dt);
		schedule();
	};
	const schedule = () => {
		if (!handle && visible && !document.hidden && !stopped) handle = requestAnimationFrame(tick);
	};
	const observer = new IntersectionObserver((entries) => {
		visible = entries.some((entry) => entry.isIntersecting);
		if (!visible) last = -1;
		schedule();
	});
	observer.observe(canvas);
	const onVisibility = () => {
		last = -1;
		schedule();
	};
	document.addEventListener('visibilitychange', onVisibility);
	schedule();
	return () => {
		stopped = true;
		if (handle) cancelAnimationFrame(handle);
		observer.disconnect();
		document.removeEventListener('visibilitychange', onVisibility);
	};
}

/** GLSL helpers shared by the shaders: an integer hash and Gaussian noise from it. */
export const GLSL_NOISE = `
uint hash_u(uint x) {
	x ^= x >> 16; x *= 0x7feb352du;
	x ^= x >> 15; x *= 0x846ca68bu;
	x ^= x >> 16;
	return x;
}
float hash_f(uvec3 p) {
	uint h = hash_u(p.x ^ hash_u(p.y ^ hash_u(p.z)));
	return (float(h >> 8) + 0.5) * (1.0 / 16777216.0);
}
// A standard normal sample for integer cell p and stream s (Box-Muller).
float gauss(ivec2 p, uint s) {
	uvec3 q = uvec3(uint(p.x), uint(p.y), s * 2u);
	float u1 = hash_f(q);
	float u2 = hash_f(q + uvec3(0u, 0u, 1u));
	return sqrt(-2.0 * log(u1)) * cos(6.28318530718 * u2);
}
`;
