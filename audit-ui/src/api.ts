import type { ExportResult, Frame, FramePage, Review, ReviewStatus, Split } from './types'

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init)
  if (!response.ok) {
    const detail: unknown = await response.json().catch(() => null)
    const message = typeof detail === 'object' && detail !== null && 'detail' in detail
      ? JSON.stringify(detail.detail) : response.statusText
    throw new Error(message)
  }
  return response.json() as Promise<T>
}

export function listFrames(split: Split, status: ReviewStatus, camera: string, offset: number): Promise<FramePage> {
  const query = new URLSearchParams({ split, status, camera, offset: String(offset), limit: '30' })
  return request<FramePage>(`/api/frames?${query}`)
}

export function getFrame(id: string): Promise<Frame> {
  return request<Frame>(`/api/frames/${encodeURIComponent(id)}`)
}

export function imageUrl(id: string): string {
  return `/api/frames/${encodeURIComponent(id)}/image`
}

export function saveReview(id: string, review: Review): Promise<Frame> {
  return request<Frame>(`/api/frames/${encodeURIComponent(id)}/review`, {
    method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(review),
  })
}

export function exportSnapshot(): Promise<ExportResult> {
  return request<ExportResult>('/api/exports', { method: 'POST' })
}

export function uploadFrame(file: File, camera: string, split: Exclude<Split, 'all'>): Promise<Frame> {
  const body = new FormData()
  body.append('file', file)
  body.append('camera', camera)
  body.append('split', split)
  return request<Frame>('/api/frames', { method: 'POST', body })
}
