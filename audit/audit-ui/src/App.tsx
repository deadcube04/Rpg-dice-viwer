import { useEffect, useRef, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useBlocker, useNavigate, useSearch } from '@tanstack/react-router'
import { Action, Notice } from '@/components/ui'
import { api } from '@/lib/api'
import type { Filters, Review } from '@/lib/contracts'
import { Queue } from '@/features/queue/Queue'
import { ReviewPanel } from '@/features/review/ReviewPanel'

export function App() {
  const filters = useSearch({ from: '__root__' })
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const [selected, setSelected] = useState<string | null>(null)
  const [dirty, setDirty] = useState(false)
  const allowNavigation = useRef(false)
  const saving = useRef(false)
  const [savingActive, setSavingActive] = useState(false)
  const [nextBusy, setNextBusy] = useState(false)
  const [notice, setNotice] = useState<string | null>(null)
  const progress = useQuery({ queryKey: ['progress'], queryFn: api.progress, refetchInterval: 5000, refetchOnWindowFocus: true })
  const queue = useQuery({ queryKey: ['queue', filters], queryFn: () => api.queue(filters) })
  const activeId = selected ?? queue.data?.items[0]?.image_id ?? null
  const frame = useQuery({ queryKey: ['frame', activeId], queryFn: () => api.frame(activeId ?? ''), enabled: activeId !== null })
  const exporting = useMutation({ mutationFn: api.export, onSuccess: result => setNotice(`Snapshot de auditoria: ${result.path}. ${result.reviewed}/${result.frames} revisadas; não aprovado para treinamento.`) })
  const backingUp = useMutation({ mutationFn: api.backup, onSuccess: result => setNotice(`Backup SQLite criado em ${result.path}`) })
  useBlocker({ shouldBlockFn: () => saving.current || (!allowNavigation.current && dirty && !window.confirm('Há alterações não salvas. Sair desta imagem? O rascunho local será preservado.')) })
  useEffect(() => {
    if (selected === null && progress.data?.last_image) setSelected(progress.data.last_image)
  }, [progress.data?.last_image])
  useEffect(() => { if (activeId) void api.resume(activeId).catch(() => setNotice('Não foi possível registrar a última imagem.')) }, [activeId])
  function select(id: string) {
    if (saving.current) return
    if (id === activeId) return
    if (dirty && !window.confirm('Há alterações não salvas. Abrir outra imagem? O rascunho será preservado.')) return
    setDirty(false); setSelected(id)
  }
  function changeFilters(next: Filters) {
    if (saving.current) return
    if (dirty && !window.confirm('Há alterações não salvas. Alterar os filtros? O rascunho será preservado.')) return
    setDirty(false); setSelected(null)
    allowNavigation.current = true
    void navigate({ to: '/', search: next }).finally(() => { allowNavigation.current = false })
  }
  async function nextInQueue() {
    if (!activeId || nextBusy || saving.current) return
    setNextBusy(true)
    setNotice(null)
    try {
      const currentItems = queue.data?.items ?? []
      const currentIndex = currentItems.findIndex(item => item.image_id === activeId)
      const nextItem = currentItems[currentIndex + 1]
      if (nextItem) { select(nextItem.image_id); return }

      const refreshed = await queue.refetch()
      const latest = refreshed.data
      if (!latest) return
      const refreshedIndex = latest.items.findIndex(item => item.image_id === activeId)
      const following = latest.items[refreshedIndex + 1] ?? (refreshedIndex < 0 ? latest.items[0] : undefined)
      if (following) { select(following.image_id); return }
      if (filters.offset + 30 < latest.total) {
        changeFilters({ ...filters, offset: filters.offset + 30 })
        return
      }
      setNotice('Fila atualizada. Não há próxima imagem neste filtro.')
    } finally { setNextBusy(false) }
  }
  async function save(review: Review, next: boolean) {
    if (!activeId) throw new Error('Selecione uma imagem')
    saving.current = true
    setSavingActive(true)
    try {
    const oldQueue = queue.data?.items ?? []
    const currentIndex = oldQueue.findIndex(item => item.image_id === activeId)
    const nextId = oldQueue[currentIndex + 1]?.image_id
    const saved = await api.save(activeId, review)
    try { localStorage.removeItem(`audit-draft-${activeId}`) }
    catch { setNotice('Revisão salva no servidor; não foi possível limpar o rascunho local.') }
    setDirty(false)
    queryClient.setQueryData(['frame', activeId], saved)
    await Promise.all([queryClient.invalidateQueries({ queryKey: ['queue'] }), queryClient.invalidateQueries({ queryKey: ['progress'] })])
    if (next) {
      if (nextId) setSelected(nextId)
      else {
        const latest = await api.queue(filters)
        const retainedIndex = latest.items.findIndex(item => item.image_id === activeId)
        const following = latest.items[retainedIndex >= 0 ? retainedIndex + 1 : Math.max(0, currentIndex)]
        if (following) setSelected(following.image_id)
        else if (filters.offset + 30 < latest.total) {
          setSelected(null)
          allowNavigation.current = true
          saving.current = false
          try { await navigate({ to: '/', search: { ...filters, offset: filters.offset + 30 } }) }
          finally { allowNavigation.current = false }
        }
        else setNotice('Fim da fila neste filtro. A revisão completa e aprovação continuam manuais.')
      }
    }
    } finally { saving.current = false; setSavingActive(false) }
  }
  return <div className="min-h-screen bg-slate-50 text-slate-900">
    <header className="flex min-h-20 flex-wrap items-center justify-between gap-3 border-b border-slate-200 bg-white px-4 py-3 lg:px-6">
      <div><p className="text-lg font-semibold tracking-tight">RPG Dice <span className="ml-2 text-sm font-normal text-slate-500">Auditoria local</span></p><p className="text-xs text-slate-500">ZIP imutável · Revisões independentes · Etapas 2–7 bloqueadas</p></div>
      <div className="flex flex-wrap gap-2"><Action busy={backingUp.isPending} onClick={() => backingUp.mutate()}>Criar backup</Action><Action busy={exporting.isPending} onClick={() => exporting.mutate()}>Exportar revisões</Action></div>
    </header>
    {notice && <div className="px-4"><Notice>{notice}</Notice><Action onClick={() => setNotice(null)}>Fechar aviso</Action></div>}
    {(exporting.isError || backingUp.isError) && <Notice error>{exporting.error?.message ?? backingUp.error?.message}</Notice>}
    {progress.isError && <Notice error>{progress.error.message}</Notice>}
    <div className="grid grid-cols-1 lg:grid-cols-[20rem_minmax(0,1fr)]">
      <Queue filters={filters} onFilters={changeFilters} items={queue.data?.items ?? []} total={queue.data?.total ?? 0} progress={progress.data}
        selected={activeId} onSelect={select} loading={queue.isPending} error={queue.isError ? queue.error.message : null} />
      <main className="min-w-0 lg:h-[calc(100vh-5rem)] lg:overflow-auto">
        {!activeId && !queue.isPending && <Notice>Selecione uma imagem ou altere os filtros.</Notice>}
        {activeId && frame.isPending && <Notice>Carregando imagem e anotações…</Notice>}
        {frame.isError && <Notice error>{frame.error.message}</Notice>}
        {frame.data && <ReviewPanel key={`${frame.data.image_id}:${frame.data.revision}`} frame={frame.data} saving={savingActive} nextBusy={nextBusy} onNext={nextInQueue} onSave={save} onDirty={setDirty} onSelect={select} />}
      </main>
    </div>
  </div>
}
