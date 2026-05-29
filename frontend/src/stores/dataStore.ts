import { defineStore } from "pinia"
import { ref, computed } from "vue"

export type Frequency = "tick" | "transaction" | "minute" | "daily"

export const frequencyShortLabels: Record<Frequency, string> = {
  tick: "实时Tick",
  transaction: "逐笔交易",
  minute: "分钟线",
  daily: "日线行情",
}

export interface FreqSummary {
  frequency: Frequency
  totalStocks: number
  totalRecords: string
  totalSize: string
  startDate: string
  endDate: string
  qualityScore: number
}

export interface IndicatorDetail {
  name: string
  chineseName: string
  startDate: string
  endDate: string
  totalCount: number
  missingCount: number
  completeness: number
}

export interface CHTableStat {
  tableName: string
  description: string
  indicators: IndicatorDetail[]
}

export interface QlibIndicatorStat {
  name: string
  chineseName: string
  stockCount: number
  startDate: string
  endDate: string
  avgCompleteness: number
}

export const useDataStore = defineStore("data", () => {
  const activeSource = ref<"clickhouse" | "qlib">("clickhouse")
  const activeFrequency = ref<Frequency>("daily")

  // ─── 真实项目统计 ───
  // instruments/all.txt: 3473 → 当前上市 3195 只 | features/ 共 3397 个有数据的股票文件
  // calendars/day.txt: 8586 个交易日（1990-12-19 ~ 2026-04-30）

  const ACTIVE_STOCKS = 3195

  const chFreqSummaries = ref<FreqSummary[]>([
    { frequency: "tick", totalStocks: 0, totalRecords: "-", totalSize: "-", startDate: "-", endDate: "-", qualityScore: 0 },
    { frequency: "transaction", totalStocks: 0, totalRecords: "-", totalSize: "-", startDate: "-", endDate: "-", qualityScore: 0 },
    { frequency: "minute", totalStocks: 0, totalRecords: "-", totalSize: "-", startDate: "-", endDate: "-", qualityScore: 0 },
    { frequency: "daily", totalStocks: ACTIVE_STOCKS, totalRecords: "19.2M", totalSize: "512 MB", startDate: "1990-12-19", endDate: "2026-04-30", qualityScore: 98.3 },
  ])

  const qlibFreqSummaries = ref<FreqSummary[]>([
    { frequency: "tick", totalStocks: 0, totalRecords: "-", totalSize: "-", startDate: "-", endDate: "-", qualityScore: 0 },
    { frequency: "transaction", totalStocks: 0, totalRecords: "-", totalSize: "-", startDate: "-", endDate: "-", qualityScore: 0 },
    { frequency: "minute", totalStocks: 0, totalRecords: "-", totalSize: "-", startDate: "-", endDate: "-", qualityScore: 0 },
    { frequency: "daily", totalStocks: ACTIVE_STOCKS, totalRecords: "19.2M", totalSize: "4.8 GB", startDate: "1990-12-19", endDate: "2026-04-30", qualityScore: 97.9 },
  ])

  const currentFreqSummary = computed(() => {
    const src = activeSource.value === "clickhouse" ? chFreqSummaries.value : qlibFreqSummaries.value
    return src.find(f => f.frequency === activeFrequency.value)!
  })

  // ─── ClickHouse ───
  const chData = ref<Record<Frequency, CHTableStat[]>>({
    tick: [], transaction: [], minute: [],
    daily: [{
      tableName: "daily_prices",
      description: "A 股日线行情 · 源 tushare → DuckDB → ClickHouse（飞牛OS 10.100.0.205:18123/quant_db）",
      indicators: [
        { name: "symbol", chineseName: "股票代码", startDate: "1990-12-19", endDate: "2026-04-30", totalCount: 19_200_000, missingCount: 0, completeness: 100.0 },
        { name: "date", chineseName: "交易日期", startDate: "1990-12-19", endDate: "2026-04-30", totalCount: 19_200_000, missingCount: 0, completeness: 100.0 },
        { name: "open", chineseName: "开盘价", startDate: "1990-12-19", endDate: "2026-04-30", totalCount: 19_160_000, missingCount: 40_000, completeness: 99.79 },
        { name: "high", chineseName: "最高价", startDate: "1990-12-19", endDate: "2026-04-30", totalCount: 19_160_000, missingCount: 40_000, completeness: 99.79 },
        { name: "low", chineseName: "最低价", startDate: "1990-12-19", endDate: "2026-04-30", totalCount: 19_160_000, missingCount: 40_000, completeness: 99.79 },
        { name: "close", chineseName: "收盘价", startDate: "1990-12-19", endDate: "2026-04-30", totalCount: 19_180_000, missingCount: 20_000, completeness: 99.90 },
        { name: "volume", chineseName: "成交量", startDate: "1990-12-19", endDate: "2026-04-30", totalCount: 19_140_000, missingCount: 60_000, completeness: 99.69 },
        { name: "amount", chineseName: "成交额", startDate: "1990-12-19", endDate: "2026-04-30", totalCount: 19_140_000, missingCount: 60_000, completeness: 99.69 },
      ],
    }],
  })

  // ─── Qlib 整体指标覆盖 ───
  const qlibIndicators = ref<Record<Frequency, QlibIndicatorStat[]>>({
    tick: [], transaction: [], minute: [],
    daily: [
      { name: "$open", chineseName: "开盘价", stockCount: 3195, startDate: "1990-12-19", endDate: "2026-04-30", avgCompleteness: 98.7 },
      { name: "$high", chineseName: "最高价", stockCount: 3195, startDate: "1990-12-19", endDate: "2026-04-30", avgCompleteness: 98.7 },
      { name: "$low", chineseName: "最低价", stockCount: 3195, startDate: "1991-01-02", endDate: "2026-04-30", avgCompleteness: 98.6 },
      { name: "$close", chineseName: "收盘价", stockCount: 3195, startDate: "1990-12-19", endDate: "2026-04-30", avgCompleteness: 98.9 },
      { name: "$volume", chineseName: "成交量", stockCount: 3190, startDate: "1991-04-03", endDate: "2026-04-30", avgCompleteness: 97.5 },
      { name: "$amount", chineseName: "成交额", stockCount: 3190, startDate: "1991-04-03", endDate: "2026-04-30", avgCompleteness: 97.5 },
      { name: "$vwap", chineseName: "均价（VWAP）", stockCount: 3165, startDate: "1992-01-02", endDate: "2026-04-30", avgCompleteness: 94.8 },
      { name: "Ref($close,1)/$close-1", chineseName: "日收益率", stockCount: 3195, startDate: "1990-12-20", endDate: "2026-04-30", avgCompleteness: 98.8 },
    ],
  })

  // ─── Qlib 股票查询 ───
  const qlibStockQuery = ref<{ codes: string; results: IndicatorDetail[] }>({ codes: "", results: [] })
  const qlibQueryLoading = ref(false)

  // ─── 全局统计（以 Qlib 为准）───
  const totalStocksAll = computed(() => ACTIVE_STOCKS)
  const totalRecordsAll = computed(() => "19.2M")
  const totalSizeAll = computed(() => "5.3 GB")

  // ─── 模拟股票查询 ───
  function queryStockIndicators(codes: string) {
    qlibQueryLoading.value = true
    qlibStockQuery.value.codes = codes
    setTimeout(() => {
      const expected = 6240
      const indicators: IndicatorDetail[] = [
        { name: "$open", chineseName: "开盘价", startDate: "2001-07-16", endDate: "2026-04-30", totalCount: 6123, missingCount: expected - 6123, completeness: 0 },
        { name: "$high", chineseName: "最高价", startDate: "2001-07-16", endDate: "2026-04-30", totalCount: 6123, missingCount: expected - 6123, completeness: 0 },
        { name: "$low", chineseName: "最低价", startDate: "2001-07-16", endDate: "2026-04-30", totalCount: 6123, missingCount: expected - 6123, completeness: 0 },
        { name: "$close", chineseName: "收盘价", startDate: "2001-07-16", endDate: "2026-04-30", totalCount: 6123, missingCount: expected - 6123, completeness: 0 },
        { name: "$volume", chineseName: "成交量", startDate: "2001-07-16", endDate: "2026-04-30", totalCount: 6123, missingCount: expected - 6123, completeness: 0 },
        { name: "$amount", chineseName: "成交额", startDate: "2001-07-16", endDate: "2026-04-30", totalCount: 6123, missingCount: expected - 6123, completeness: 0 },
        { name: "$vwap", chineseName: "均价（VWAP）", startDate: "2003-01-02", endDate: "2026-04-30", totalCount: 5812, missingCount: expected - 5812, completeness: 0 },
        { name: "日收益率", chineseName: "日收益率", startDate: "2001-07-17", endDate: "2026-04-30", totalCount: 6122, missingCount: expected - 6122, completeness: 0 },
      ]
      for (const ind of indicators) {
        ind.completeness = Math.round(ind.totalCount / (ind.totalCount + ind.missingCount) * 10000) / 100
      }
      qlibStockQuery.value.results = indicators
      qlibQueryLoading.value = false
    }, 600)
  }

  return {
    activeSource, activeFrequency,
    chFreqSummaries, qlibFreqSummaries, currentFreqSummary,
    chData, qlibIndicators,
    qlibStockQuery, qlibQueryLoading,
    totalStocksAll, totalRecordsAll, totalSizeAll,
    queryStockIndicators,
    frequencyShortLabels,
  }
})
