<template>
  <div>
    <!-- === Page Header === -->
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:16px">
      <div>
        <n-h2 style="margin:0 0 4px 0">数据管理</n-h2>
        <n-p style="margin:0;color:#8b949e;font-size:13px">
          {{ dataStore.activeSource === "clickhouse" ? "ClickHouse（飞牛OS 192.168.x.x）" : "Qlib（本地项目 qlib_data/）" }}
        </n-p>
      </div>
      <n-grid :x-gap="20" :cols="4" style="min-width:520px">
        <n-grid-item><n-statistic label="覆盖股票" :value="dataStore.totalStocksAll.toLocaleString()" /></n-grid-item>
        <n-grid-item><n-statistic label="总记录数" :value="dataStore.totalRecordsAll" /></n-grid-item>
        <n-grid-item><n-statistic label="存量大小" :value="dataStore.totalSizeAll" /></n-grid-item>
        <n-grid-item><n-statistic label="数据源" value="ClickHouse + Qlib" /></n-grid-item>
      </n-grid>
    </div>

    <!-- === Data Source Tabs === -->
    <n-tabs v-model:value="dataStore.activeSource" type="line" animated>
      <n-tab-pane name="clickhouse" tab="ClickHouse（飞牛OS）">
        <SourceContent source="clickhouse" />
      </n-tab-pane>
      <n-tab-pane name="qlib" tab="Qlib（本地项目）">
        <SourceContent source="qlib" />
      </n-tab-pane>
    </n-tabs>
  </div>
</template>

<script setup lang="ts">
import { useDataStore } from "@/stores/dataStore"
import SourceContent from "./SourceContent.vue"

const dataStore = useDataStore()
</script>
