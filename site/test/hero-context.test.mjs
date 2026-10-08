// The hero renders only on a GPU: a browser that would emulate one on the CPU gets the
// still frame, since these shaders run there at a few frames a second and block the page.

import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createContext } from '../src/hero/gl.ts';

function canvas(renderer) {
	const requested = [];
	const lost = [];
	const gl = {
		RENDERER: 0x1f01,
		getExtension(name) {
			if (name === 'WEBGL_debug_renderer_info') return { UNMASKED_RENDERER_WEBGL: 0x9246 };
			if (name === 'WEBGL_lose_context') return { loseContext: () => lost.push(name) };
			return name === 'EXT_color_buffer_float' ? {} : null;
		},
		getParameter: () => renderer,
	};
	return { requested, lost, element: { getContext: (kind, attributes) => (requested.push(attributes), gl) } };
}

for (const renderer of ['Google SwiftShader', 'llvmpipe (LLVM 17.0.6, 256 bits)']) {
	test(`a ${renderer} renderer gets the still frame and its context is released`, () => {
		const { requested, lost, element } = canvas(renderer);
		assert.equal(createContext(element), null);
		assert.equal(lost.length, 1);
		assert.equal(requested[0].failIfMajorPerformanceCaveat, true);
	});
}

test('a hardware renderer gets a context', () => {
	const { lost, element } = canvas('ANGLE (NVIDIA, NVIDIA GeForce RTX 4080)');
	assert.notEqual(createContext(element), null);
	assert.equal(lost.length, 0);
});
