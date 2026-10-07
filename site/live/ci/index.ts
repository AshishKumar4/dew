import { operatorAuthorized, remoteRun } from '../src/remote';
export { RemoteJob, RunnerCache, RunnerFleet } from '../src/remote';
export default {
	async fetch(request: Request, env: Parameters<typeof remoteRun>[1]) {
		if (!operatorAuthorized(request, env)) return new Response('Forbidden', { status: 403 });
		try { return await remoteRun(request, env); }
		catch (error) { return Response.json({ message: String(error) }, { status: 400 }); }
	},
};
