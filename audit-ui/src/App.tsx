import { useState } from 'react'
import { keepPreviousData, useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { exportSnapshot, getFrame, listFrames, saveReview, uploadFrame } from './api'
import { ReviewEditor } from './ReviewEditor'
import { UploadPanel } from './UploadPanel'
import type { ExportResult, ReviewStatus, Split } from './types'

const SPLITS: { value: Split; label: string }[] = [
  { value: 'train', label: 'Treino' }, { value: 'valid', label: 'Validação' },
  { value: 'test', label: 'Teste' }, { value: 'all', label: 'Todas' },
]

export function App() {
  const queryClient = useQueryClient()
  const [split, setSplit] = useState<Split>('train')
  const [status, setStatus] = useState<ReviewStatus>('pending')
  const [camera, setCamera] = useState('')
  const [offset, setOffset] = useState(0)
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [exported, setExported] = useState<ExportResult | null>(null)

  const listing = useQuery({ queryKey: ['frames', split, status, camera, offset],
    queryFn: () => listFrames(split, status, camera, offset), placeholderData: keepPreviousData })
  const activeId = selectedId ?? listing.data?.items[0]?.id ?? null
  const detail = useQuery({ queryKey: ['frame', activeId],
    queryFn: () => getFrame(activeId ?? ''), enabled: activeId !== null })
  const save = useMutation({ mutationFn: (review: Parameters<typeof saveReview>[1]) => saveReview(activeId ?? '', review),
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ['frames'] })
      await queryClient.invalidateQueries({ queryKey: ['frame', activeId] })
      if (status === 'pending') setSelectedId(null)
    },
  })
  const exporting = useMutation({ mutationFn: exportSnapshot, onSuccess: setExported })
  const uploading = useMutation({ mutationFn: ({ file, camera, targetSplit }: {
    file: File; camera: string; targetSplit: Exclude<Split, 'all'>
  }) => uploadFrame(file, camera, targetSplit), onSuccess: async frame => {
    setSplit(frame.split)
    setStatus('all')
    setCamera('')
    setOffset(0)
    setSelectedId(frame.id)
    await queryClient.invalidateQueries({ queryKey: ['frames'] })
  } })

  function changeSplit(value: Split) { setSplit(value); setOffset(0); setSelectedId(null) }
  function changeStatus(value: ReviewStatus) { setStatus(value); setOffset(0); setSelectedId(null) }

  return <div className="app-shell">
    <header className="topbar"><div className="brand-mark">◆</div><div className="brand-copy"><strong>RPG Dice</strong><span>Auditoria de dados</span></div>
      <div className="topbar-right"><span className="local-indicator"><span /> LOCAL</span>
        <button type="button" className="export-button" onClick={() => exporting.mutate()} disabled={exporting.isPending}>
          {exporting.isPending ? 'Exportando…' : 'Exportar snapshot'} <span aria-hidden="true">↗</span></button></div></header>
    <div className="app-body"><aside className="queue-panel" aria-label="Fila de imagens">
      <div className="queue-head"><div className="eyebrow">CURADORIA / DATASET</div><h1>Fila de revisão</h1>
        <p>Confira as caixas e o valor da face antes de incluir no próximo treino.</p></div>
      <div className="queue-filters"><div className="tabs" role="group" aria-label="Partição">
        {SPLITS.map(option => <button type="button" key={option.value} className={split === option.value ? 'active' : ''}
          onClick={() => changeSplit(option.value)}>{option.label}</button>)}</div>
        <div className="filter-row"><label>Estado<select value={status} onChange={event => changeStatus(event.target.value as ReviewStatus)}>
          <option value="pending">Pendentes</option><option value="reviewed">Revisadas</option><option value="all">Todas</option></select></label>
          <label>Câmera<select value={camera} onChange={event => { setCamera(event.target.value); setOffset(0); setSelectedId(null) }}>
            <option value="">Todas</option>{listing.data?.cameras.map(item => <option key={item} value={item}>{item}</option>)}</select></label></div></div>
      <UploadPanel busy={uploading.isPending} error={uploading.isError ? uploading.error.message : null}
        onUpload={(file, cameraName, targetSplit) => uploading.mutate({ file, camera: cameraName, targetSplit })} />
      <div className="queue-count">{listing.isPending ? 'Carregando imagens…' : `${listing.data?.total ?? 0} imagens`}<span>{offset + 1}–{Math.min(offset + 30, listing.data?.total ?? 0)}</span></div>
      <div className="queue-list">
        {listing.isError && <div className="queue-state error" role="alert">Não foi possível carregar a fila: {listing.error.message}</div>}
        {listing.isSuccess && listing.data.items.length === 0 && <div className="queue-state">Nenhuma imagem neste filtro.</div>}
        {listing.data?.items.map(item => <button type="button" key={item.id}
          className={`queue-item ${activeId === item.id ? 'selected' : ''}`}
          onClick={() => setSelectedId(item.id)}>
          <img src={`/api/frames/${encodeURIComponent(item.id)}/image`} alt="" loading="lazy" />
          <span className="queue-item-copy"><strong>{item.id}</strong><small>{item.camera}</small><small>{item.dice.length} {item.dice.length === 1 ? 'dado' : 'dados'} · {item.epoch}</small></span>
          <span className={`status-dot ${item.reviewed ? 'done' : ''}`} aria-label={item.reviewed ? 'Revisada' : 'Pendente'} /></button>)}
      </div>
      <div className="pagination"><button type="button" disabled={offset === 0} onClick={() => { setOffset(Math.max(0, offset - 30)); setSelectedId(null) }}>← Anterior</button>
        <button type="button" disabled={offset + 30 >= (listing.data?.total ?? 0)}
          onClick={() => { setOffset(offset + 30); setSelectedId(null) }}>Próximas →</button></div>
    </aside>
    <main className="main-panel">
      {detail.isPending && <div className="main-state">Carregando imagem…</div>}
      {detail.isError && <div className="main-state error" role="alert">Não foi possível abrir a imagem: {detail.error.message}</div>}
      {detail.isSuccess && <ReviewEditor key={detail.data.id} frame={detail.data} saving={save.isPending} onSave={review => save.mutate(review)} />}
      {save.isError && <div className="toast error" role="alert">Erro ao salvar: {save.error.message}</div>}
      {save.isSuccess && <div className="toast" role="status">Revisão salva.</div>}
      {exporting.isError && <div className="toast error" role="alert">Erro ao exportar: {exporting.error.message}</div>}
      {exported && <div className="export-panel" role="status"><button type="button" aria-label="Fechar aviso" onClick={() => setExported(null)}>×</button>
        <strong>Snapshot criado</strong><span>{exported.reviewed} imagens revisadas · {exported.frames} fotos</span>
        <code>{exported.path}</code><small>Execute <code>uv run dice-data prepare --raw "{exported.path}" --out data/processed-next</code> para preparar o retreino.</small></div>}
    </main></div>
  </div>
}
