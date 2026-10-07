export type Split = 'all' | 'train' | 'valid' | 'test'
export type ReviewStatus = 'all' | 'pending' | 'reviewed'
export type DieType = 'd4' | 'd6' | 'd8' | 'd10' | 'd12' | 'd20'
export type Box = { x: number; y: number; w: number; h: number }
export type Die = { type: DieType; value: number | null; box: Box }
export type Frame = {
  id: string
  file_name: string
  camera: string
  epoch: string
  width: number
  height: number
  split: Exclude<Split, 'all'>
  reviewed: boolean
  values_trusted: boolean
  note: string
  dice: Die[]
}
export type FramePage = { items: Frame[]; total: number; offset: number; cameras: string[] }
export type Review = Pick<Frame, 'dice' | 'values_trusted' | 'note'>
export type ExportResult = { path: string; frames: number; reviewed: number; manifest_sha256: string }
