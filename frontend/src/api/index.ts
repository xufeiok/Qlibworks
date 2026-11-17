import axios from "axios"

const api = axios.create({
  baseURL: "/api/v1",
  timeout: 30000,
  headers: { "Content-Type": "application/json" },
})

api.interceptors.response.use(
  (res) => res.data,
  (err) => {
    console.error("API Error:", err)
    return Promise.reject(err)
  }
)

export default api
