<template>
  <div>
    <n-h2>回测管理</n-h2>
    <n-p>Backtrader 回测引擎 | SuperPlot 可视化分析</n-p>
    <n-grid :x-gap="12" :y-gap="12" :cols="4">
      <n-grid-item v-for="i in 4" :key="i">
        <n-card :title="'回测 ' + i" hoverable>
          <n-skeleton text :repeat="3" />
        </n-card>
      </n-grid-item>
    </n-grid>
    <n-empty description="正在开发中..." style="margin-top:24px" />
  </div>
</template>
<script setup lang="ts">
</script>
