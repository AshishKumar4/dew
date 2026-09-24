// Who a request comes from, for the per-visitor limits. An IPv4 address is one
// visitor. An IPv6 visitor is its /64: a home or cloud network is routinely handed a
// whole /64, so keying on the full address would give one visitor 2^64 identities.

/** The eight 16-bit groups of an IPv6 address, or null when `address` is not one. */
function ipv6Groups(address: string): number[] | null {
	const halves = address.split('::');
	if (halves.length > 2) return null;
	const parse = (part: string): number[] | null => {
		if (part === '') return [];
		const groups: number[] = [];
		for (const group of part.split(':')) {
			if (!/^[0-9a-f]{1,4}$/i.test(group)) return null;
			groups.push(Number.parseInt(group, 16));
		}
		return groups;
	};
	const head = parse(halves[0]);
	const tail = halves.length === 2 ? parse(halves[1]) : [];
	if (head === null || tail === null) return null;
	if (halves.length === 1) return head.length === 8 ? head : null;
	const zeros = 8 - head.length - tail.length;
	return zeros >= 1 ? [...head, ...new Array<number>(zeros).fill(0), ...tail] : null;
}

/**
 * The key the per-visitor limits count against: an IPv4 address as it is, an
 * IPv4-mapped IPv6 address as its IPv4 address, any other IPv6 address as its /64.
 */
export function visitorKey(ip: string): string {
	const mapped = /^::ffff:(\d{1,3}(?:\.\d{1,3}){3})$/i.exec(ip);
	if (mapped) return mapped[1];
	if (!ip.includes(':')) return ip;
	// A zone index (fe80::1%eth0) names an interface, not a network.
	const groups = ipv6Groups(ip.split('%')[0]);
	if (groups === null) return ip;
	return `${groups
		.slice(0, 4)
		.map((group) => group.toString(16))
		.join(':')}::/64`;
}
