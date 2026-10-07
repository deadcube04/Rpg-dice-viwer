import { useState } from 'react'
import type { Split } from './types'

type Props = {
  busy: boolean
  error: string | null
  onUpload: (file: File, camera: string, split: Exclude<Split, 'all'>) => void
}

export function UploadPanel({ busy, error, onUpload }: Props) {
  const [open, setOpen] = useState(false)
  const [file, setFile] = useState<File | null>(null)
  const [camera, setCamera] = useState('')
  const [split, setSplit] = useState<Exclude<Split, 'all'>>('train')

  return <div className="upload-panel">
    <button type="button" className="upload-toggle" aria-expanded={open} onClick={() => setOpen(current => !current)}>
      <span>+ Nova foto</span><span>{open ? '−' : '+'}</span>
    </button>
    {open && <form onSubmit={event => {
      event.preventDefault()
      if (file && camera.trim()) onUpload(file, camera.trim(), split)
    }}>
      <label>Imagem JPEG ou PNG<input type="file" accept="image/jpeg,image/png" required
        onChange={event => setFile(event.currentTarget.files?.[0] ?? null)} /></label>
      <label>Câmera<input value={camera} maxLength={100} required placeholder="Ex.: iPhone pessoal"
        onChange={event => setCamera(event.target.value)} /></label>
      <label>Partição<select value={split} onChange={event => setSplit(event.target.value as Exclude<Split, 'all'>)}>
        <option value="train">Treino</option><option value="valid">Validação</option><option value="test">Teste reservado</option>
      </select></label>
      <p>Reserve fotos novas para validação e teste. Cada foto fica em uma única partição.</p>
      {error && <p className="upload-error" role="alert">{error}</p>}
      <button type="submit" className="upload-submit" disabled={busy || !file || !camera.trim()}>{busy ? 'Enviando…' : 'Adicionar à fila'}</button>
    </form>}
  </div>
}
