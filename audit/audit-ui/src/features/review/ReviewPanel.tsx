import { useEffect, useState } from 'react'
import { useForm } from '@tanstack/react-form'
import { useQuery } from '@tanstack/react-query'
import { Action, Field, Notice, RadioChoices } from '@/components/ui'
import { api, imageUrl } from '@/lib/api'
import { frameSchema } from '@/lib/contracts'
import type { Frame, Review } from '@/lib/contracts'
import { ImageEditor } from '@/features/review/ImageEditor'
import { InstanceEditor } from '@/features/review/InstanceEditor'

const reviewSchema = frameSchema.pick({ revision: true, decision: true, reason: true, note: true, instances: true })
function initialReview(frame: Frame): Review {
  const base = { revision: frame.revision, decision: frame.decision, reason: frame.reason,
    note: frame.note, instances: frame.instances }
  try {
    const draft = reviewSchema.safeParse(JSON.parse(localStorage.getItem(`audit-draft-${frame.image_id}`) ?? 'null'))
    return draft.success ? draft.data : base
  } catch { return base }
}
export function ReviewPanel({ frame, saving, nextBusy, onNext, onSave, onDirty, onSelect }: {
  frame: Frame; saving: boolean; nextBusy: boolean; onNext: () => Promise<void>; onSave: (review: Review, next: boolean) => Promise<void>; onDirty: (dirty: boolean) => void; onSelect: (id: string) => void
}) {
  const [initial] = useState(() => initialReview(frame))
  const [dirty, setDirty] = useState(() => initial.revision !== frame.revision || JSON.stringify(initial.instances) !== JSON.stringify(frame.instances) || initial.decision !== frame.decision || initial.note !== frame.note || initial.reason !== frame.reason)
  const [selected, setSelected] = useState<string | null>(initial.instances[0]?.instance_id ?? null)
  const [localBusy, setBusy] = useState(false)
  const busy = saving || localBusy
  const [error, setError] = useState<string | null>(null)
  const [saved, setSaved] = useState(false)
  const form = useForm({ defaultValues: initial })
  const history = useQuery({ queryKey: ['history', frame.image_id, frame.revision], queryFn: () => api.history(frame.image_id) })
  const duplicates = useQuery({ queryKey: ['duplicates', frame.image_id], queryFn: () => api.duplicates(frame.image_id) })
  function persist() {
    setDirty(true); setSaved(false)
    try { localStorage.setItem(`audit-draft-${frame.image_id}`, JSON.stringify(form.state.values)) }
    catch { setError('Não foi possível guardar o rascunho local. Salve antes de sair.') }
  }
  async function save(next: boolean) {
    if (busy) return
    setBusy(true); setError(null)
    try {
      await onSave(form.state.values, next)
      setDirty(false); setSaved(true)
    } catch (cause) { setError(cause instanceof Error ? cause.message : 'Falha ao salvar') }
    finally { setBusy(false) }
  }
  useEffect(() => { onDirty(dirty) }, [dirty, onDirty])
  useEffect(() => {
    const prevent = (event: BeforeUnloadEvent) => { if (dirty) { event.preventDefault(); event.returnValue = '' } }
    window.addEventListener('beforeunload', prevent)
    return () => window.removeEventListener('beforeunload', prevent)
  }, [dirty])
  useEffect(() => {
    const shortcut = (event: KeyboardEvent) => { if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 's') { event.preventDefault(); void save(event.shiftKey) } }
    window.addEventListener('keydown', shortcut)
    return () => window.removeEventListener('keydown', shortcut)
  })
  return <section className="min-w-0">
    <header className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-200 bg-white p-4">
      <div className="min-w-0"><h2 className="break-all text-sm font-semibold">{frame.path}</h2>
        <p className="mt-1 text-xs text-slate-600">Imagem: <code>{frame.image_id}</code> · Revisão {frame.revision}</p>
        <p className="mt-1 text-xs text-slate-500">{frame.annotation_source ?? 'Sem anotação final'} · {dirty ? 'Alterações não salvas' : 'Sem alterações locais'}</p></div>
      <div className="flex flex-wrap gap-2"><Action busy={busy} onClick={() => { void save(false) }}>Salvar</Action><Action primary busy={busy} onClick={() => { void save(true) }}>Salvar e próxima</Action><Action disabled={busy || nextBusy} onClick={() => { void onNext() }}>PRÓXIMA</Action></div>
    </header>
    {error && <Notice error>{error} {error.startsWith('409') && 'O rascunho permanece neste navegador. Recarregue e concilie as versões pelo histórico.'}</Notice>}
    {saved && <Notice>Revisão salva. Isso não aprova o dataset nem inicia a próxima etapa.</Notice>}
    <form.Subscribe selector={state => state.values}>{values => {
      const item = values.instances.find(candidate => candidate.instance_id === selected)
      const updateInstances = (instances: Review['instances']) => { form.setFieldValue('instances', instances); persist() }
      return <div className="grid min-w-0 grid-cols-1 xl:grid-cols-[minmax(0,1fr)_19rem]">
        <ImageEditor frame={frame} instances={values.instances} selected={selected} onSelect={setSelected} onChange={updateInstances} disabled={busy} />
        <div className="flex min-w-0 flex-col gap-3 bg-white p-4">
          {values.revision !== frame.revision && <Notice error>Rascunho preservado da revisão {values.revision}; servidor na revisão {frame.revision}. Compare com o histórico antes de conciliar. Salvar permanece sujeito a conflito.</Notice>}
          {values.revision !== frame.revision && <Action disabled={busy} onClick={() => {
            if (window.confirm('Após comparar o histórico, usar este rascunho sobre a revisão atual? A próxima gravação criará uma nova revisão e preservará o histórico.')) {
              form.setFieldValue('revision', frame.revision); persist()
            }
          }}>Conciliar sobre revisão atual</Action>}
          <RadioChoices label="Decisão da imagem" name="image-decision" value={values.decision} options={[{ value: 'pending', label: 'Pendente' }, { value: 'reviewed', label: 'Revisada' }, { value: 'negative', label: 'Sem dados' }, { value: 'excluded', label: 'Excluída' }]}
            disabled={busy} onChange={value => { const decision = reviewSchema.shape.decision.parse(value); form.setFieldValue('decision', decision); persist() }} />
          <Field label="Motivo da exclusão / decisão" value={values.reason} onChange={v => { form.setFieldValue('reason', v); persist() }} disabled={busy} />
          <Field label="Observações" multiline value={values.note} onChange={v => { form.setFieldValue('note', v); persist() }} disabled={busy} />
          <p className="text-xs text-slate-500">Pendente guarda rascunho; Revisada confirma a conferência; Sem dados exige zero instâncias; Excluída exige motivo.</p>
          <h3 className="text-sm font-semibold">Instâncias ({values.instances.length})</h3>
          <div className="flex flex-wrap gap-1">{values.instances.map((candidate, index) => <Action key={candidate.instance_id} label={`Selecionar instância ${index + 1}`} pressed={selected === candidate.instance_id} primary={selected === candidate.instance_id} onClick={() => setSelected(candidate.instance_id)}>{index + 1} · {candidate.die_type ?? '?'}</Action>)}</div>
          <p aria-live="polite" className="text-xs text-slate-600">{item ? `Instância selecionada: ${values.instances.findIndex(candidate => candidate.instance_id === selected) + 1} de ${values.instances.length}` : 'Nenhuma instância selecionada'}</p>
          {item ? <InstanceEditor item={item} disabled={busy} onChange={changed => updateInstances(values.instances.map(candidate => candidate.instance_id === changed.instance_id ? changed : candidate))}
            onRemove={() => { updateInstances(values.instances.filter(candidate => candidate.instance_id !== selected)); setSelected(null) }} /> : <Notice>Selecione uma instância ou use “Criar caixa”.</Notice>}
        </div>
      </div>
    }}</form.Subscribe>
    <div className="grid gap-4 border-t border-slate-200 p-4 md:grid-cols-2">
      <details><summary className="cursor-pointer text-sm font-semibold">Histórico de revisões ({history.data?.length ?? 0})</summary>
        {history.isError && <Notice error>{history.error.message}</Notice>}
        {history.data?.map(row => <div key={row.revision} className="mt-2 flex flex-wrap items-center justify-between gap-2 border-b border-slate-200 pb-2 text-xs"><span>r{row.revision} · {row.created_at} · {row.document.decision}</span>
          <Action disabled={busy} onClick={() => { form.setFieldValue('instances', row.document.instances); form.setFieldValue('decision', row.document.decision); form.setFieldValue('reason', row.document.reason); form.setFieldValue('note', row.document.note); persist() }}>Restaurar como rascunho</Action></div>)}</details>
      <details><summary className="cursor-pointer text-sm font-semibold">Duplicatas exatas sugeridas ({duplicates.data?.length ?? 0})</summary>
        <p className="mt-2 text-xs text-slate-600">Mesmo hash não confirma a anotação. Compare e revise cada entrada.</p>
        {duplicates.isError && <Notice error>{duplicates.error.message}</Notice>}
        {duplicates.data?.map(row => <div key={row.image_id} className="mt-2 flex items-center gap-2"><img className="h-20 w-24 object-contain" src={imageUrl(row.image_id)} alt={`Comparação: ${row.path}`} /><Action disabled={busy} onClick={() => onSelect(row.image_id)}>Abrir {row.image_id.slice(0, 8)}</Action></div>)}</details>
    </div>
    <details className="px-4 pb-4 text-xs text-slate-600"><summary className="cursor-pointer">Guia e atalhos</summary><p className="mt-2">Ctrl/Cmd+S: salvar. Ctrl/Cmd+Shift+S: salvar e próxima. Caixas devem cobrir o corpo visível, sem sombra. Numeral não equivale a corpo. Registre convenção de d4, d10 e ambiguidade 6/9; não invente valor. Coordenadas permitem edição por teclado. Nenhuma revisão anterior conta como nova confirmação.</p></details>
  </section>
}
