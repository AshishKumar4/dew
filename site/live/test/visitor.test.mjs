// Per-visitor limits key an IPv6 address by its /64.
//
//   pnpm live:test

import assert from 'node:assert/strict';
import { test } from 'node:test';
import { visitorKey } from '../src/visitor.ts';

test('addresses in one /64 are one visitor, however they are written', () => {
	const key = visitorKey('2606:4700:3032:0:1:2:3:4');
	assert.equal(key, '2606:4700:3032:0::/64');
	assert.equal(visitorKey('2606:4700:3032::ac43:d6ba'), key);
	assert.equal(visitorKey('2606:4700:3032:0000:ffff:ffff:ffff:ffff'), key);
	assert.equal(visitorKey('2606:4700:3032::'), key);
});

test('different /64s are different visitors', () => {
	assert.notEqual(visitorKey('2606:4700:3032:1::1'), visitorKey('2606:4700:3032:2::1'));
	assert.notEqual(visitorKey('::1'), visitorKey('1::1'));
});

test('an IPv4 address is its own visitor, also when mapped into IPv6', () => {
	assert.equal(visitorKey('203.0.113.7'), '203.0.113.7');
	assert.equal(visitorKey('::ffff:203.0.113.7'), '203.0.113.7');
	assert.notEqual(visitorKey('203.0.113.7'), visitorKey('203.0.113.8'));
});

test('a zone index does not make a new visitor', () => {
	assert.equal(visitorKey('fe80::1%eth0'), visitorKey('fe80::2'));
});

test('a string that is not an address is kept whole', () => {
	assert.equal(visitorKey('unknown'), 'unknown');
	assert.equal(visitorKey('1:2:3'), '1:2:3');
	assert.equal(visitorKey('1::2::3'), '1::2::3');
});
