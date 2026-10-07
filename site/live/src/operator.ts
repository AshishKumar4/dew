export function operatorAuthorized(request: Request, env: { OPERATOR_SECRET: string }): boolean {
	const expected = new TextEncoder().encode(`Bearer ${env.OPERATOR_SECRET}`);
	const supplied = new TextEncoder().encode(request.headers.get('Authorization') ?? '');
	return Boolean(env.OPERATOR_SECRET) && expected.length === supplied.length && crypto.subtle.timingSafeEqual(expected, supplied);
}
