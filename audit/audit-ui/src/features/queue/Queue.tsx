import { Action, Choice, Field, Meter, Notice } from '@/components/ui'
import { dieTypes } from '@/lib/contracts'
import { imageUrl } from '@/lib/api'
import type { Filters, Frame, Progress } from '@/lib/contracts'

const all = { value: '', label: 'Todos' }
type StringFilterKey = Exclude<keyof Filters, 'double_check' | 'offset'>
export function Queue({ filters, onFilters, items, total, progress, selected, onSelect, loading, error }: {
  filters: Filters; onFilters: (filters: Filters) => void; items: Frame[]; total: number;
  progress: Progress | undefined; selected: string | null; onSelect: (id: string) => void; loading: boolean; error: string | null
}) {
  const change = (key: StringFilterKey, value: string) => onFilters({ ...filters, [key]: value, offset: 0 })
  const changeDoubleCheck = (enabled: boolean) => onFilters({ ...filters, double_check: enabled,
    decision: enabled ? '' : filters.decision || 'pending', offset: 0 })
  return <aside className="flex min-w-0 flex-col gap-4 border-r border-slate-200 bg-white p-4 lg:h-[calc(100vh-5rem)] lg:overflow-auto" aria-label="Fila de revisão">
    <div><p className="text-xs font-semibold uppercase tracking-widest text-teal-700">Etapa 01 / Curadoria</p>
      <h1 className="mt-1 text-2xl font-semibold">Auditoria integral</h1>
      <p className="mt-2 text-sm text-slate-600">Cada imagem exige sua decisão, inclusive duplicatas.</p></div>
    {progress && <div><p className="text-xs font-medium text-slate-600">Progresso global do escopo ativo</p><p className="text-sm">{progress.reviewed.toLocaleString('pt-BR')} / {progress.total.toLocaleString('pt-BR')} revisadas</p><Meter complete={progress.reviewed} total={progress.total} />
      <p className="mb-2 text-xs text-slate-500">Atualizado a cada 5 segundos. Os filtros não alteram este total.</p>
      {progress.ignored_sources.length > 0 && <p className="mb-2 break-all text-xs text-slate-500">Ignorada por enquanto: {progress.ignored_sources.join(', ')}. Registros preservados.</p>}
      <details className="text-xs"><summary className="cursor-pointer">Progresso por fonte</summary>{Object.entries(progress.sources).map(([source, count]) => <p key={source} className="mt-2 break-all">{source}: {count.reviewed}/{count.total}</p>)}</details></div>}
    <Field label="Buscar identificador ou caminho" value={filters.search} onChange={v => change('search', v)} />
    <div className="grid grid-cols-2 gap-2">
      <Choice label="Estado" value={filters.decision} options={[filters.double_check && !filters.decision ? { value: '', label: 'Pendentes e revisadas' } : all, { value: 'pending', label: 'Pendente' }, { value: 'reviewed', label: 'Revisada' }, { value: 'negative', label: 'Sem dados' }, { value: 'excluded', label: 'Excluída' }]} onChange={v => change('decision', v)} />
      <Choice label="Tipo" value={filters.type} options={[all, ...dieTypes.map(value => ({ value, label: value }))]} onChange={v => change('type', v)} />
      <Choice label="Origem" value={filters.source} options={[all, ...Object.keys(progress?.sources ?? {}).map(value => ({ value, label: value }))]} onChange={v => change('source', v)} />
      <Choice label="Contagem anotada" value={filters.count} options={[all, ...['0', '1', '2', '3', '4', '5', '6', '7', '8', '9+'].map(value => ({ value, label: value }))]} onChange={v => change('count', v)} />
    </div>
    <Choice label="Problema de origem" value={filters.issue} options={[all, ...(progress?.issues ?? []).map(value => ({ value, label: value }))]} onChange={v => change('issue', v)} />
    <Choice label="Double check" value={filters.double_check ? 'on' : ''} options={[{ value: '', label: 'Não' }, { value: 'on', label: 'Sim' }]} onChange={v => changeDoubleCheck(v === 'on')} />
    {filters.double_check && <p className="-mt-3 text-xs text-slate-500">Inclui pendentes e revisadas; use Estado para restringir.</p>}
    <div className="flex items-center justify-between text-xs text-slate-500"><span>{total} neste filtro</span><span>{total ? filters.offset + 1 : 0}–{Math.min(filters.offset + 30, total)}</span></div>
    {loading && <Notice>Carregando fila…</Notice>}{error && <Notice error>{error}</Notice>}
    {!loading && !error && items.length === 0 && <Notice>Nenhuma imagem neste filtro.</Notice>}
    <div className="flex flex-col gap-1">{items.map(item => <button type="button" key={item.image_id} onClick={() => onSelect(item.image_id)} aria-current={selected === item.image_id ? 'true' : undefined}
      className={`flex items-center gap-3 rounded border p-2 text-left focus-visible:outline-2 focus-visible:outline-teal-700 ${selected === item.image_id ? 'border-teal-700 bg-teal-50' : 'border-transparent hover:bg-slate-50'}`}>
      <img className="h-12 w-16 shrink-0 rounded bg-slate-100 object-contain" src={imageUrl(item.image_id)} alt="" loading="lazy" />
      <span className="min-w-0"><span className="block truncate text-xs font-semibold">{item.path.split('/').at(-1)}</span><span className="text-xs text-slate-500">{item.instances.length} objetos · {item.decision}</span></span></button>)}</div>
    <div className="flex flex-wrap gap-2"><Action disabled={!filters.offset} onClick={() => onFilters({ ...filters, offset: Math.max(0, filters.offset - 30) })}>Anterior</Action>
      <Action disabled={filters.offset + 30 >= total} onClick={() => onFilters({ ...filters, offset: filters.offset + 30 })}>Próximas</Action></div>
  </aside>
}
