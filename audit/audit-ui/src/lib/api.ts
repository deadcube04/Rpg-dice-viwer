import { z } from 'zod'
import { exportSchema, frameSchema, historySchema, pageSchema, progressSchema } from '@/lib/contracts'
import type { Filters, Review } from '@/lib/contracts'

async function request<T>(path: string, schema: z.ZodType<T>, method = 'GET', body?: unknown): Promise<T> {
  const response = await fetch(`/api/${path}`, { method,
    ...(body === undefined ? {} : { headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }) })
  const data: unknown = await response.json()
  if (!response.ok) {
    const error = z.object({ detail: z.string() }).safeParse(data)
    throw new Error(`${response.status}: ${error.success ? error.data.detail : 'Falha na API'}`)
  }
  return schema.parse(data)
}
export const api = {
  progress: () => request('progress', progressSchema),
  queue: (filters: Filters) => request(`frames?${new URLSearchParams(Object.entries(filters).map(([k, v]) => [k, String(v)]))}`, pageSchema),
  frame: (id: string) => request(`frames/${id}`, frameSchema),
  save: (id: string, review: Review) => request(`frames/${id}/review`, frameSchema, 'PUT', review),
  resume: (id: string) => request(`frames/${id}/resume`, z.object({ saved: z.boolean() }), 'POST', {}),
  history: (id: string) => request(`frames/${id}/history`, historySchema),
  duplicates: (id: string) => request(`frames/${id}/duplicates`, z.array(frameSchema)),
  export: () => request('exports', exportSchema, 'POST', {}),
  backup: () => request('backup', z.object({ path: z.string() }), 'POST', {}),
}
export const imageUrl = (id: string) => `/api/frames/${id}/image`
