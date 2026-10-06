import { DurableObject } from 'cloudflare:workers';
import { RunnerFleet, remoteRun } from '../src/remote';
import { commandOf } from '../src/remote-plan';
export { RunnerFleet };

export class Job extends DurableObject<Env> {
	async expire() {
		if (this.env.FAIL_STOP) throw new Error('container destruction failed');
		await this.ctx.storage.put('expired', true);
	}
	async expired() { return (await this.ctx.storage.get('expired')) ?? false; }
}

interface Env {
	RUNNER_FLEET: DurableObjectNamespace<RunnerFleet>;
	REMOTE_JOB: DurableObjectNamespace<Job>;
	RUNNER_SECRET: string;
	FAIL_STOP?: boolean;
}

export default {
	async fetch(request: Request, env: Env) {
		const scenario = new URL(request.url).pathname.slice(1);
		if (scenario === 'auth' || scenario === 'large' || scenario === 'null') {
			const body = scenario === 'large' ? 'x'.repeat(600_001) : 'null';
			return remoteRun(new Request('https://test/v1/remote/run', {
				method: 'POST', body, headers: scenario === 'auth' ? {} : { Authorization: `Bearer ${env.RUNNER_SECRET}` },
			}), env as never);
		}
		if (scenario === 'command') {
			const refused = [[], ['a\0b'], Array(129).fill('x'), [1]].map((value) => {
				try { commandOf(value); return false; } catch { return true; }
			});
			return Response.json({ refused, accepted: commandOf(['python', '-c', 'print("hello")']) });
		}
		const fleet = env.RUNNER_FLEET.get(env.RUNNER_FLEET.idFromName('global'));
		const jobs = (await Promise.all(Array.from({ length: 20 }, () => fleet.acquire(1000)))).filter(Boolean) as string[];
		if (scenario === 'cap') return Response.json({ jobs });
		try { await fleet.acquire(1000 + 40 * 60_000); } catch {}
		return Response.json({ jobs, expired: await Promise.all(jobs.map((id) =>
			env.REMOTE_JOB.get(env.REMOTE_JOB.idFromName(id)).expired())) });
	},
};
