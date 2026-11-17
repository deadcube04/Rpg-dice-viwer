import { Action, Choice, Field, Flag } from '@/components/ui'
import { dieTypes, normalize, tokens } from '@/lib/contracts'
import type { Instance } from '@/lib/contracts'

export function InstanceEditor({ item, onChange, onRemove, disabled }: { item: Instance; onChange: (item: Instance) => void; onRemove: () => void; disabled: boolean }) {
  const update = (change: Partial<Instance>) => onChange({ ...item, ...change })
  return <section className="flex flex-col gap-3 border-t border-slate-200 pt-3" aria-label="Anotação do objeto">
    <Choice label="Tipo do dado" value={item.die_type ?? ''} options={[{ value: '', label: 'Desconhecido' }, ...dieTypes.map(value => ({ value, label: value }))]} disabled={disabled}
      onChange={value => { const type = dieTypes.find(candidate => candidate === value) ?? null; update({ die_type: type, type_known: type !== null }) }} />
    <Choice label="Token textual da face" value={item.face_token ?? ''} options={[{ value: '', label: 'Ilegível / desconhecido' }, ...tokens.map(value => ({ value, label: value }))]} disabled={disabled}
      onChange={value => update({ face_token: value || null, face_value: normalize(value || null), value_known: value !== '' && !item.result_ambiguous })} />
    <p className="text-xs text-slate-600">Valor normalizado: <strong>{normalize(item.face_token) ?? '—'}</strong>. Tokens 0, 00 e 10 permanecem distintos.</p>
    {typeof item.proposed_token === 'string' && <p className="text-xs text-amber-800">Proposta antiga não confirmada: {item.proposed_token}. Selecione o token correto após conferir a imagem.</p>}
    <Flag label="Valor confirmado nesta instância" checked={item.value_known} disabled={disabled || !item.face_token || item.result_ambiguous} onChange={value_known => update({ value_known })} />
    <Flag label="Resultado ambíguo" checked={item.result_ambiguous} disabled={disabled} onChange={result_ambiguous => update({ result_ambiguous, ...(result_ambiguous ? { value_known: false } : {}) })} />
    <Flag label="Oclusão" checked={item.occluded} disabled={disabled} onChange={occluded => update({ occluded })} />
    <Flag label="Corte na borda" checked={item.truncated} disabled={disabled} onChange={truncated => update({ truncated })} />
    <Choice label="Região anotada" value={item.box_kind} options={[{ value: 'body', label: 'Corpo do dado' }, { value: 'numeral', label: 'Somente numeral' }]} disabled={disabled}
      onChange={value => update({ box_kind: value === 'body' ? 'body' : 'numeral', detection_known: value === 'body' })} />
    <Choice label="Modalidade da face" value={item.modality} options={[{ value: 'unknown', label: 'Não identificada' }, { value: 'pips', label: 'Pontos (d6)' }, { value: 'numerals', label: 'Numerais' }]} disabled={disabled}
      onChange={value => update({ modality: value === 'pips' ? 'pips' : value === 'numerals' ? 'numerals' : 'unknown' })} />
    <Field label="Convenção d4/d10 ou coleção" value={item.convention} onChange={convention => update({ convention })} disabled={disabled} />
    <div className="grid grid-cols-2 gap-2">{(['x1', 'y1', 'x2', 'y2'] as const).map((label, index) => <Field key={label} label={`${label} (pixels)`} value={String(Math.round(item.bbox_xyxy[index] * 100) / 100)} disabled={disabled}
      onChange={value => { const number = Number(value); if (!Number.isFinite(number)) return; const box: Instance['bbox_xyxy'] = [...item.bbox_xyxy]; box[index] = number; update({ bbox_xyxy: box, polygon: null }) }} />)}</div>
    <Action danger onClick={onRemove} disabled={disabled}>Remover instância</Action>
  </section>
}
