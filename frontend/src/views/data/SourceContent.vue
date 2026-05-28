<template>
  <div>
    <!-- Frequency Selector -->
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:16px">
      <n-button-group size="small">
        <n-button
          v-for="f in frequencies"
          :key="f.key"
          :type="dataStore.activeFrequency === f.key ? 'primary' : 'default'"
          :disabled="f.disabled"
          @click="selectFreq(f.key)"
        >
          {{ f.label }}
        </n-button>
      </n-button-group>
      <n-tag v-if="!hasData" size="small" type="warning">暂无数据</n-tag>
    </div>

    <!-- Frequency Stats -->
    <template v-if="hasData">
      <!-- Summary Cards -->
      <n-grid :x-gap="12" :y-gap="12" :cols="5" style="margin-bottom:16px">
        <n-grid-item>
          <n-card :style="{ background: '#161b22', border: '1px solid #30363d' }" size="small">
            <n-statistic label="覆盖股票" :value="summary.totalStocks.toLocaleString()" />
          </n-card>
        </n-grid-item>
        <n-grid-item>
          <n-card :style="{ background: '#161b22', border: '1px solid #30363d' }" size="small">
            <n-statistic label="总记录数" :value="summary.totalRecords" />
          </n-card>
        </n-grid-item>
        <n-grid-item>
          <n-card :style="{ background: '#161b22', border: '1px solid #30363d' }" size="small">
            <n-statistic label="存量大小" :value="summary.totalSize" />
          </n-card>
        </n-grid-item>
        <n-grid-item>
          <n-card :style="{ background: '#161b22', border: '1px solid #30363d' }" size="small">
            <n-statistic label="时间范围" :value="summary.startDate">
              <template #suffix><n-text depth="3" style="font-size:11px">~ {{ summary.endDate }}</n-text></template>
            </n-statistic>
          </n-card>
        </n-grid-item>
        <n-grid-item>
          <n-card :style="{ background: '#161b22', border: '1px solid #30363d' }" size="small">
            <n-statistic label="数据质量">
              <template #suffix><n-text depth="3" style="font-size:11px">分</n-text></template>
            </n-statistic>
            <n-progress type="line" :percentage="summary.qualityScore" :height="6" :border-radius="3" style="margin-top:2px" />
          </n-card>
        </n-grid-item>
      </n-grid>

      <!-- ClickHouse: Table Structure -->
      <template v-if="source === 'clickhouse'">
        <n-card
          title="表结构 · 指标详情"
          :style="{ background: '#161b22', border: '1px solid #30363d' }"
          size="small"
          style="margin-bottom:16px"
        >
          <template #header-extra>
            <n-tag size="small" type="info">{{ chTables.length }} 张表</n-tag>
          </template>
          <n-collapse accordion>
            <n-collapse-item v-for="tbl in chTables" :key="tbl.tableName" :title="tbl.tableName" :name="tbl.tableName">
              <template #header-extra>
                <n-text depth="3" style="font-size:12px">{{ tbl.description }}</n-text>
              </template>
              <n-data-table
                :columns="chIndicatorColumns"
                :data="tbl.indicators"
                :bordered="false"
                size="small"
                :single-line="false"
                :style="{ background: 'transparent' }"
              />
            </n-collapse-item>
          </n-collapse>
        </n-card>
      </template>

      <!-- Qlib: Overall Structure + Per-Stock Query -->
      <template v-if="source === 'qlib'">
        <n-card
          title="整体数据结构 · 指标覆盖"
          :style="{ background: '#161b22', border: '1px solid #30363d' }"
          size="small"
          style="margin-bottom:16px"
        >
          <template #header-extra>
            <n-tag size="small" type="info">{{ qlibIndicators.length }} 个指标</n-tag>
          </template>
          <n-data-table
            :columns="qlibIndicatorColumns"
            :data="qlibIndicators"
            :bordered="false"
            size="small"
            :single-line="false"
            :style="{ background: 'transparent' }"
          />
        </n-card>

        <n-card
          title="股票数据查询"
          :style="{ background: '#161b22', border: '1px solid #30363d' }"
          size="small"
          style="margin-bottom:16px"
        >
          <template #header-extra>
            <n-tag v-if="dataStore.qlibStockQuery.codes" size="small" type="warning">{{ dataStore.qlibStockQuery.codes }}</n-tag>
          </template>
          <div style="display:flex;gap:12px;margin-bottom:16px">
            <n-input
              v-model:value="queryCode"
              placeholder="输入股票代码，多个用逗号分隔（如 000001.SZ, 600519.SH）"
              clearable
              :style="{ minWidth: '420px' }"
              @keyup.enter="doQuery"
            />
            <n-button type="primary" :loading="dataStore.qlibQueryLoading" @click="doQuery" :disabled="!queryCode.trim()">
              查询
            </n-button>
          </div>
          <n-data-table
            v-if="dataStore.qlibStockQuery.results.length > 0"
            :columns="stockIndicatorColumns"
            :data="dataStore.qlibStockQuery.results"
            :bordered="false"
            size="small"
            :single-line="false"
            :style="{ background: 'transparent' }"
          />
          <n-empty v-else description="输入股票代码后点击查询查看各指标详情" />
        </n-card>
      </template>
    </template>

    <!-- No data placeholder -->
    <template v-else>
      <n-card :style="{ background: '#161b22', border: '1px solid #30363d' }" size="small">
        <n-empty description="该频率暂无数据接入，请先配置数据同步任务" />
      </n-card>
    </template>
  </div>
</template>

<script setup lang="ts">
import { h, ref, computed } from "vue"
import { useDataStore, type Frequency, type FreqSummary, type QlibIndicatorStat } from "@/stores/dataStore"

const props = defineProps<{ source: "clickhouse" | "qlib" }>()
const dataStore = useDataStore()
const queryCode = ref("")

const freq = computed(() => dataStore.activeFrequency)

const frequencies = computed(() => [
  { key: "tick" as Frequency, label: "实时Tick", disabled: !hasFreqData("tick") },
  { key: "transaction" as Frequency, label: "逐笔交易", disabled: !hasFreqData("transaction") },
  { key: "minute" as Frequency, label: "分钟线", disabled: !hasFreqData("minute") },
  { key: "daily" as Frequency, label: "日线行情", disabled: !hasFreqData("daily") },
])

function hasFreqData(freq: Frequency): boolean {
  if (props.source === "clickhouse") {
    return dataStore.chData[freq].length > 0
  } else {
    return dataStore.qlibIndicators[freq].length > 0
  }
}

const hasData = computed(() => hasFreqData(freq.value))

const summary = computed(() => {
  const src = props.source === "clickhouse" ? dataStore.chFreqSummaries : dataStore.qlibFreqSummaries
  return src.find(f => f.frequency === freq.value)!
})

const chTables = computed(() => dataStore.chData[freq.value])
const qlibIndicators = computed(() => dataStore.qlibIndicators[freq.value])

function selectFreq(f: Frequency) {
  dataStore.activeFrequency = f
}

// ─── Render helpers ───
function renderProgress(pct: number) {
  const color = pct >= 95 ? "#39d2c0" : pct >= 85 ? "#eab308" : "#ef4444"
  return h("div", { style: "display:flex;align-items:center;gap:8px" }, [
    h("div", { style: "flex:1" }, h("n-progress", { type: "line", percentage: pct, height: 4, "border-radius": 2, indicatorPlacement: "inside", color })),
    h("span", { style: "font-size:12px;color:#8b949e;min-width:42px;text-align:right" }, pct.toFixed(1) + "%"),
  ])
}

function renderMissing(missing: number) {
  const color = missing === 0 ? "#8b949e" : missing < 10000 ? "#eab308" : "#ef4444"
  return h("span", { style: `font-size:12px;color:${color}` }, missing === 0 ? "-" : missing.toLocaleString())
}

function renderCount(count: number) {
  return h("span", { style: "font-size:12px" }, count.toLocaleString())
}

// ─── ClickHouse 指标列 ───
const chIndicatorColumns = [
  { title: "指标名", key: "name", width: 110 },
  { title: "中文名", key: "chineseName", width: 110 },
  { title: "起始时间", key: "startDate", width: 120 },
  { title: "截至时间", key: "endDate", width: 120 },
  { title: "总数", key: "totalCount", width: 110, render: (_: any, row: any) => renderCount(row.totalCount) },
  { title: "缺少数", key: "missingCount", width: 100, render: (_: any, row: any) => renderMissing(row.missingCount) },
  { title: "完整度", key: "completeness", width: 180, render: (_: any, row: any) => renderProgress(row.completeness) },
]

// ─── Qlib 指标覆盖列 ───
const qlibIndicatorColumns = [
  { title: "指标名", key: "name", width: 100 },
  { title: "中文名", key: "chineseName", width: 100 },
  { title: "覆盖股票", key: "stockCount", width: 100, render: (_: any, row: any) => row.stockCount.toLocaleString() },
  { title: "最早起始", key: "startDate", width: 120 },
  { title: "最迟截至", key: "endDate", width: 120 },
  { title: "平均完整度", key: "avgCompleteness", width: 180, render: (_: any, row: any) => renderProgress(row.avgCompleteness) },
]

// ─── 股票指标详情列 ───
const stockIndicatorColumns = [
  { title: "指标名", key: "name", width: 100 },
  { title: "中文名", key: "chineseName", width: 100 },
  { title: "起始时间", key: "startDate", width: 120 },
  { title: "截至时间", key: "endDate", width: 120 },
  { title: "总数", key: "totalCount", width: 100, render: (_: any, row: any) => renderCount(row.totalCount) },
  { title: "缺少数", key: "missingCount", width: 100, render: (_: any, row: any) => renderMissing(row.missingCount) },
  { title: "完整度", key: "completeness", width: 180, render: (_: any, row: any) => renderProgress(row.completeness) },
]

// ─── 查询 ───
function doQuery() {
  if (!queryCode.value.trim()) return
  dataStore.queryStockIndicators(queryCode.value.trim())
}
</script>




