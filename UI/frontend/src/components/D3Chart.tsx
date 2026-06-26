import { useEffect, useRef } from 'react'
import * as d3 from 'd3'
import type { D3Spec } from '../api/client'

/**
 * Generic renderer for the framework-agnostic `d3-buck/1` specs emitted by
 * UI/backend/d3_viz.py. Supports line / multiline / area / bar / heatmap marks
 * with axes, a tooltip, and (for time-series marks) horizontal zoom + pan.
 *
 * The same specs are what the `visualize_training` MCP tool returns to Claude;
 * here we turn them into an interactive SVG for the human in the loop.
 */

interface Props {
  spec: D3Spec | null
  height?: number
}

const MARGIN = { top: 16, right: 20, bottom: 40, left: 56 }

export default function D3Chart({ spec, height = 320 }: Props) {
  const ref = useRef<SVGSVGElement | null>(null)
  const wrapRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    if (!ref.current || !wrapRef.current || !spec) return
    const svgEl = ref.current
    const width = wrapRef.current.clientWidth || 640

    const svg = d3.select(svgEl)
    svg.selectAll('*').remove()
    svg.attr('width', width).attr('height', height)

    const data = spec.data ?? []
    if (data.length === 0) {
      svg
        .append('text')
        .attr('x', width / 2)
        .attr('y', height / 2)
        .attr('text-anchor', 'middle')
        .attr('fill', '#9ca3af')
        .attr('font-size', 13)
        .text((spec.meta?.reason as string) || 'No data for this chart')
      return
    }

    const enc = spec.encoding as any
    const innerW = width - MARGIN.left - MARGIN.right
    const innerH = height - MARGIN.top - MARGIN.bottom
    const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`)

    // ── tooltip ──
    let tip = d3.select(wrapRef.current).select<HTMLDivElement>('.d3-tooltip')
    if (tip.empty()) {
      tip = d3
        .select(wrapRef.current)
        .append('div')
        .attr('class', 'd3-tooltip')
        .style('position', 'absolute')
        .style('pointer-events', 'none')
        .style('background', 'rgba(17,24,39,0.95)')
        .style('color', '#f9fafb')
        .style('padding', '4px 8px')
        .style('border-radius', '4px')
        .style('font-size', '11px')
        .style('opacity', '0')
    }
    const showTip = (html: string, ev: MouseEvent) => {
      const [mx, my] = d3.pointer(ev, wrapRef.current as any)
      tip.html(html).style('left', `${mx + 12}px`).style('top', `${my}px`).style('opacity', '1')
    }
    const hideTip = () => tip.style('opacity', '0')

    if (spec.mark === 'bar') {
      const x = d3
        .scaleBand<string>()
        .domain(data.map((d) => String((d as any)[enc.x.field])))
        .range([0, innerW])
        .padding(0.1)
      const yMax = d3.max(data, (d) => +(d as any)[enc.y.field]) ?? 1
      const y = d3.scaleLinear().domain([0, yMax]).nice().range([innerH, 0])
      const color = (enc.series?.[0]?.color as string) || '#0ea5e9'

      g.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(x).tickValues(x.domain().filter((_, i) => i % Math.ceil(data.length / 10) === 0)))
      g.append('g').call(d3.axisLeft(y))
      g.selectAll('rect')
        .data(data)
        .join('rect')
        .attr('x', (d) => x(String((d as any)[enc.x.field])) ?? 0)
        .attr('y', (d) => y(+(d as any)[enc.y.field]))
        .attr('width', x.bandwidth())
        .attr('height', (d) => innerH - y(+(d as any)[enc.y.field]))
        .attr('fill', color)
        .on('mousemove', (ev, d) => showTip(`${enc.x.field}: ${(d as any)[enc.x.field]}<br/>${enc.y.field}: ${(d as any)[enc.y.field]}`, ev))
        .on('mouseleave', hideTip)
      drawAxisLabels(g, enc, innerW, innerH)
      return
    }

    if (spec.mark === 'heatmap') {
      const xField = enc.x.field
      const intensity = enc.intensity
      const x = d3
        .scaleBand<string>()
        .domain(data.map((d) => String((d as any)[xField])))
        .range([0, innerW])
        .padding(0.02)
      const vals = data.map((d) => +(d as any)[intensity.field])
      const domain = (intensity.domain as [number, number]) || [d3.min(vals) ?? 0, d3.max(vals) ?? 1]
      const interp = intensity.scheme === 'rdylgn' ? d3.interpolateRdYlGn : d3.interpolateViridis
      const color = d3.scaleSequential(interp).domain(domain as [number, number])

      g.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(x).tickValues(x.domain().filter((_, i) => i % Math.ceil(data.length / 12) === 0)))
      g.selectAll('rect')
        .data(data)
        .join('rect')
        .attr('x', (d) => x(String((d as any)[xField])) ?? 0)
        .attr('y', 0)
        .attr('width', x.bandwidth())
        .attr('height', innerH)
        .attr('fill', (d) => color(+(d as any)[intensity.field]))
        .on('mousemove', (ev, d) => showTip(`${xField}: ${(d as any)[xField]}<br/>${intensity.field}: ${(d as any)[intensity.field]}`, ev))
        .on('mouseleave', hideTip)
      drawAxisLabels(g, { x: enc.x, y: { title: intensity.title } }, innerW, innerH)
      return
    }

    // ── line / multiline / area ──
    const xField = enc.x.field
    const series: { field: string; label: string; color: string }[] =
      enc.series ?? [{ field: enc.y.field, label: enc.y.title ?? enc.y.field, color: '#6366f1' }]

    const xExtent = d3.extent(data, (d) => +(d as any)[xField]) as [number, number]
    const x = d3.scaleLinear().domain(xExtent).range([0, innerW])
    const allY: number[] = []
    series.forEach((s) => data.forEach((d) => {
      const v = (d as any)[s.field]
      if (v !== null && v !== undefined && !Number.isNaN(+v)) allY.push(+v)
    }))
    const y = d3
      .scaleLinear()
      .domain([Math.min(0, d3.min(allY) ?? 0), d3.max(allY) ?? 1])
      .nice()
      .range([innerH, 0])

    const xAxisG = g.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(x))
    g.append('g').call(d3.axisLeft(y))
    drawAxisLabels(g, enc, innerW, innerH)

    // clip for zoom
    const clipId = `clip-${Math.random().toString(36).slice(2)}`
    g.append('clipPath').attr('id', clipId).append('rect').attr('width', innerW).attr('height', innerH)
    const plot = g.append('g').attr('clip-path', `url(#${clipId})`)

    const drawSeries = (xs: d3.ScaleLinear<number, number>) => {
      plot.selectAll('.series').remove()
      series.forEach((s) => {
        const pts = data.filter((d) => (d as any)[s.field] !== null && (d as any)[s.field] !== undefined)
        if (spec.mark === 'area' && series.length === 1) {
          const area = d3
            .area<any>()
            .x((d) => xs(+d[xField]))
            .y0(y(Math.max(0, y.domain()[0])))
            .y1((d) => y(+d[s.field]))
          plot.append('path').attr('class', 'series').datum(pts).attr('fill', s.color).attr('opacity', 0.25).attr('d', area as any)
        }
        const line = d3
          .line<any>()
          .defined((d) => d[s.field] !== null && d[s.field] !== undefined)
          .x((d) => xs(+d[xField]))
          .y((d) => y(+d[s.field]))
        plot
          .append('path')
          .attr('class', 'series')
          .datum(pts)
          .attr('fill', 'none')
          .attr('stroke', s.color)
          .attr('stroke-width', 1.6)
          .attr('d', line as any)
      })
    }
    drawSeries(x)

    // legend
    if (series.length > 1) {
      const legend = g.append('g').attr('transform', `translate(${innerW - 120},0)`)
      series.forEach((s, i) => {
        const row = legend.append('g').attr('transform', `translate(0,${i * 16})`)
        row.append('rect').attr('width', 10).attr('height', 10).attr('fill', s.color)
        row.append('text').attr('x', 14).attr('y', 9).attr('font-size', 11).attr('fill', '#374151').text(s.label)
      })
    }

    // hover guideline
    const focus = g.append('line').attr('y1', 0).attr('y2', innerH).attr('stroke', '#9ca3af').attr('stroke-dasharray', '3,3').style('opacity', 0)
    g.append('rect')
      .attr('width', innerW)
      .attr('height', innerH)
      .attr('fill', 'none')
      .attr('pointer-events', 'all')
      .on('mousemove', function (ev) {
        const [mx] = d3.pointer(ev)
        const xv = x.invert(mx)
        focus.attr('x1', mx).attr('x2', mx).style('opacity', 1)
        const nearest = data.reduce((a, b) => (Math.abs(+(a as any)[xField] - xv) < Math.abs(+(b as any)[xField] - xv) ? a : b))
        const rows = series.map((s) => `${s.label}: ${(nearest as any)[s.field]}`).join('<br/>')
        showTip(`${xField}: ${(nearest as any)[xField]}<br/>${rows}`, ev as any)
      })
      .on('mouseleave', () => {
        focus.style('opacity', 0)
        hideTip()
      })

    // zoom (x only)
    const zoom = d3
      .zoom<SVGRectElement, unknown>()
      .scaleExtent([1, 20])
      .translateExtent([[0, 0], [innerW, innerH]])
      .extent([[0, 0], [innerW, innerH]])
      .on('zoom', (ev) => {
        const zx = ev.transform.rescaleX(x)
        xAxisG.call(d3.axisBottom(zx) as any)
        drawSeries(zx)
      })
    svg.select<SVGRectElement>('rect') // noop to satisfy types
    g.select<SVGRectElement>('rect').call(zoom as any)
  }, [spec, height])

  return (
    <div ref={wrapRef} style={{ position: 'relative', width: '100%' }}>
      <svg ref={ref} />
    </div>
  )
}

function drawAxisLabels(g: any, enc: any, innerW: number, innerH: number) {
  if (enc.x?.title) {
    g.append('text')
      .attr('x', innerW / 2)
      .attr('y', innerH + 34)
      .attr('text-anchor', 'middle')
      .attr('font-size', 11)
      .attr('fill', '#6b7280')
      .text(enc.x.title)
  }
  if (enc.y?.title) {
    g.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -innerH / 2)
      .attr('y', -44)
      .attr('text-anchor', 'middle')
      .attr('font-size', 11)
      .attr('fill', '#6b7280')
      .text(enc.y.title)
  }
}
