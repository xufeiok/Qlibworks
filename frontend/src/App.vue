<template>
  <n-config-provider :theme="darkTheme" :theme-overrides="themeOverrides" :locale="zhCN" :date-locale="dateZhCN">
    <n-message-provider><n-notification-provider>
      <router-view />
    </n-notification-provider></n-message-provider>
  </n-config-provider>
</template>

<script setup lang="ts">
import { darkTheme, zhCN, dateZhCN } from "naive-ui"
import type { GlobalThemeOverrides } from "naive-ui"

const themeOverrides: GlobalThemeOverrides = {
  common: {
    primaryColor: "#39d2c0",
    primaryColorHover: "#4de0ce",
    primaryColorPressed: "#2bb8a6",
    bodyColor: "#0d1117",
    cardColor: "#161b22",
    modalColor: "#161b22",
    tableColor: "#161b22",
    inputColor: "#21262d",
    popoverColor: "#21262d",
    hoverColor: "rgba(57,210,192,0.08)",
  }
}
</script>
