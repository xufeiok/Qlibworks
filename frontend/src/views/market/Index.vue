<template>
  <div>
    <n-h2>市场行情</n-h2>
    <n-p>TradingView K线图 | AI 智能分析</n-p>
    <n-grid :x-gap="12" :y-gap="12" :cols="4">
      <n-grid-item v-for="i in 4" :key="i">
        <n-card :title="'板块 ' + i" hoverable>
          <n-skeleton text :repeat="3" />
        </n-card>
      </n-grid-item>
    </n-grid>
    <n-empty description="正在开发中..." style="margin-top:24px" />
  </div>
</template>
<script setup lang="ts">
</script>
