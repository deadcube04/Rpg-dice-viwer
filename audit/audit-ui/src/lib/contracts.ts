import { z } from 'zod'

export const dieTypes = ['d4', 'd6', 'd8', 'd10', 'd10_tens', 'd12', 'd20'] as const
export const tokens = ['0', '00', ...Array.from({ length: 20 }, (_, index) => String(index + 1)), '30', '40', '50', '60', '70', '80', '90']
export const instanceSchema = z.object({
  instance_id: z.string(), bbox_xyxy: z.tuple([z.number(), z.number(), z.number(), z.number()]),
  box_kind: z.enum(['body', 'numeral']), polygon: z.array(z.array(z.number())).nullable(),
  die_type: z.enum(dieTypes).nullable(), face_token: z.string().nullable(), face_value: z.number().nullable(),
  type_known: z.boolean(), value_known: z.boolean(), detection_known: z.boolean(),
  result_ambiguous: z.boolean(), occluded: z.boolean(), truncated: z.boolean(),
  modality: z.enum(['unknown', 'pips', 'numerals']), convention: z.string(),
}).passthrough()
export const frameSchema = z.object({
  image_id: z.string(), path: z.string(), source: z.string(), sha256: z.string(),
  width: z.number(), height: z.number(), revision: z.number(), instances: z.array(instanceSchema),
  decision: z.enum(['pending', 'reviewed', 'negative', 'excluded']), reason: z.string(), note: z.string(),
  annotation_source: z.string().nullable(), issues: z.array(z.string()),
}).passthrough()
export const pageSchema = z.object({ items: z.array(frameSchema), total: z.number(), offset: z.number() })
export const progressSchema = z.object({ total: z.number(), reviewed: z.number(),
  inventory_total: z.number(), ignored_sources: z.array(z.string()),
  sources: z.record(z.string(), z.object({ total: z.number(), reviewed: z.number() })),
  last_image: z.string().nullable(), issues: z.array(z.string()), approved: z.literal(false) })
export const historySchema = z.array(z.object({ revision: z.number(), created_at: z.string(), document: frameSchema }))
export const exportSchema = z.object({ path: z.string(), frames: z.number(), reviewed: z.number(), manifest_sha256: z.string(), approved: z.literal(false) })
export type Instance = z.infer<typeof instanceSchema>
export type Frame = z.infer<typeof frameSchema>
export type Progress = z.infer<typeof progressSchema>
export type Review = Pick<Frame, 'revision' | 'decision' | 'reason' | 'note' | 'instances'>
export type Filters = { source: string; decision: string; type: string; count: string; issue: string; search: string; double_check: boolean; offset: number }
export const defaults: Filters = { source: '', decision: 'pending', type: '', count: '', issue: '', search: '', double_check: false, offset: 0 }
export const normalize = (token: string | null): number | null => token === null ? null : ['0', '00'].includes(token) ? 10 : Number(token)
