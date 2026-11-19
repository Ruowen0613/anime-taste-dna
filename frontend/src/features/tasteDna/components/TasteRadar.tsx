import {
  PolarAngleAxis,
  PolarGrid,
  PolarRadiusAxis,
  Radar,
  RadarChart,
  ResponsiveContainer,
  Tooltip,
} from 'recharts'
import type { TasteDimension } from '../types'

type TasteRadarProps = {
  dimensions: TasteDimension[]
}

export function TasteRadar({ dimensions }: TasteRadarProps) {
  return (
    <div className="h-[320px] min-w-0 w-full sm:h-[380px]" aria-label="Taste DNA genre affinity radar chart">
      <ResponsiveContainer width="100%" height="100%">
        <RadarChart
          data={dimensions}
          cx="50%"
          cy="50%"
          outerRadius="62%"
          margin={{ top: 18, right: 24, bottom: 18, left: 24 }}
        >
          <PolarGrid stroke="rgba(255,255,255,0.14)" />
          <PolarAngleAxis dataKey="label" tick={{ fill: '#cbd5e1', fontSize: 12 }} />
          <PolarRadiusAxis angle={90} domain={[0, 100]} tickCount={6} tick={{ fill: '#64748b', fontSize: 10 }} axisLine={false} />
          <Tooltip
            contentStyle={{ background: '#151a2b', border: '1px solid rgba(255,255,255,0.14)', borderRadius: 12 }}
            itemStyle={{ color: '#a78bfa' }}
            formatter={(value) => [`${Number(value).toFixed(1)}/100`, 'Affinity']}
          />
          <Radar dataKey="score" name="Affinity" stroke="#a78bfa" fill="#f472b6" fillOpacity={0.3} strokeWidth={2} />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  )
}
