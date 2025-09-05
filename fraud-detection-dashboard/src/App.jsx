import { useState, useEffect } from 'react'
import { Button } from '@/components/ui/button.jsx'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card.jsx'
import { Input } from '@/components/ui/input.jsx'
import { Label } from '@/components/ui/label.jsx'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select.jsx'
import { Badge } from '@/components/ui/badge.jsx'
import { Alert, AlertDescription } from '@/components/ui/alert.jsx'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, PieChart, Pie, Cell } from 'recharts'
import { Shield, AlertTriangle, CheckCircle, Activity, TrendingUp, Clock, DollarSign } from 'lucide-react'
import './App.css'

function App() {
  const [transactionData, setTransactionData] = useState({
    amount: '',
    merchant_category: '',
    hour: '',
    day_of_week: '',
    is_weekend: false
  })
  
  const [predictionResult, setPredictionResult] = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  const [recentTransactions, setRecentTransactions] = useState([])
  const [systemMetrics, setSystemMetrics] = useState({
    totalTransactions: 15847,
    fraudDetected: 234,
    accuracy: 95.8,
    avgResponseTime: 87
  })

  // Mock data for charts
  const [fraudTrends, setFraudTrends] = useState([
    { time: '00:00', fraudCount: 12, totalCount: 450 },
    { time: '04:00', fraudCount: 8, totalCount: 320 },
    { time: '08:00', fraudCount: 25, totalCount: 890 },
    { time: '12:00', fraudCount: 45, totalCount: 1200 },
    { time: '16:00', fraudCount: 38, totalCount: 1100 },
    { time: '20:00', fraudCount: 28, totalCount: 950 }
  ])

  const [categoryData, setCategoryData] = useState([
    { name: 'Grocery', value: 35, color: '#8884d8' },
    { name: 'Gas Station', value: 25, color: '#82ca9d' },
    { name: 'Restaurant', value: 20, color: '#ffc658' },
    { name: 'Online', value: 15, color: '#ff7300' },
    { name: 'Other', value: 5, color: '#00ff00' }
  ])

  const merchantCategories = [
    'grocery', 'gas_station', 'restaurant', 'online', 'retail', 
    'entertainment', 'travel', 'healthcare', 'education', 'other'
  ]

  const handleInputChange = (field, value) => {
    setTransactionData(prev => ({
      ...prev,
      [field]: value
    }))
  }

  const handlePrediction = async () => {
    setIsLoading(true)
    try {
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          amount: parseFloat(transactionData.amount),
          merchant_category: transactionData.merchant_category,
          hour: parseInt(transactionData.hour),
          day_of_week: parseInt(transactionData.day_of_week),
          is_weekend: transactionData.is_weekend
        })
      })
      
      const result = await response.json()
      setPredictionResult(result)
      
      // Add to recent transactions
      const newTransaction = {
        id: Date.now(),
        ...transactionData,
        prediction: result.prediction,
        confidence: result.confidence,
        timestamp: new Date().toLocaleTimeString()
      }
      setRecentTransactions(prev => [newTransaction, ...prev.slice(0, 9)])
      
    } catch (error) {
      console.error('Prediction error:', error)
      setPredictionResult({ error: 'Failed to get prediction' })
    }
    setIsLoading(false)
  }

  const resetForm = () => {
    setTransactionData({
      amount: '',
      merchant_category: '',
      hour: '',
      day_of_week: '',
      is_weekend: false
    })
    setPredictionResult(null)
  }

  // Simulate real-time updates
  useEffect(() => {
    const interval = setInterval(() => {
      setSystemMetrics(prev => ({
        ...prev,
        totalTransactions: prev.totalTransactions + Math.floor(Math.random() * 5),
        fraudDetected: prev.fraudDetected + (Math.random() > 0.95 ? 1 : 0),
        avgResponseTime: 80 + Math.floor(Math.random() * 20)
      }))
    }, 5000)

    return () => clearInterval(interval)
  }, [])

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-2">
            <Shield className="h-8 w-8 text-blue-600" />
            <h1 className="text-3xl font-bold text-gray-900">Fraud Detection Dashboard</h1>
          </div>
          <p className="text-gray-600">Real-time fraud detection and monitoring system</p>
        </div>

        {/* System Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <Card>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600">Total Transactions</p>
                  <p className="text-2xl font-bold text-gray-900">{systemMetrics.totalTransactions.toLocaleString()}</p>
                </div>
                <Activity className="h-8 w-8 text-blue-600" />
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600">Fraud Detected</p>
                  <p className="text-2xl font-bold text-red-600">{systemMetrics.fraudDetected}</p>
                </div>
                <AlertTriangle className="h-8 w-8 text-red-600" />
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600">Model Accuracy</p>
                  <p className="text-2xl font-bold text-green-600">{systemMetrics.accuracy}%</p>
                </div>
                <TrendingUp className="h-8 w-8 text-green-600" />
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600">Avg Response Time</p>
                  <p className="text-2xl font-bold text-blue-600">{systemMetrics.avgResponseTime}ms</p>
                </div>
                <Clock className="h-8 w-8 text-blue-600" />
              </div>
            </CardContent>
          </Card>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Transaction Input Form */}
          <div className="lg:col-span-1">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <DollarSign className="h-5 w-5" />
                  Test Transaction
                </CardTitle>
                <CardDescription>
                  Enter transaction details to test fraud detection
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <Label htmlFor="amount">Transaction Amount ($)</Label>
                  <Input
                    id="amount"
                    type="number"
                    placeholder="100.00"
                    value={transactionData.amount}
                    onChange={(e) => handleInputChange('amount', e.target.value)}
                  />
                </div>

                <div>
                  <Label htmlFor="category">Merchant Category</Label>
                  <Select onValueChange={(value) => handleInputChange('merchant_category', value)}>
                    <SelectTrigger>
                      <SelectValue placeholder="Select category" />
                    </SelectTrigger>
                    <SelectContent>
                      {merchantCategories.map(category => (
                        <SelectItem key={category} value={category}>
                          {category.replace('_', ' ').toUpperCase()}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                <div>
                  <Label htmlFor="hour">Hour of Day (0-23)</Label>
                  <Input
                    id="hour"
                    type="number"
                    min="0"
                    max="23"
                    placeholder="14"
                    value={transactionData.hour}
                    onChange={(e) => handleInputChange('hour', e.target.value)}
                  />
                </div>

                <div>
                  <Label htmlFor="day">Day of Week (0-6)</Label>
                  <Input
                    id="day"
                    type="number"
                    min="0"
                    max="6"
                    placeholder="1 (Monday)"
                    value={transactionData.day_of_week}
                    onChange={(e) => handleInputChange('day_of_week', e.target.value)}
                  />
                </div>

                <div className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    id="weekend"
                    checked={transactionData.is_weekend}
                    onChange={(e) => handleInputChange('is_weekend', e.target.checked)}
                    className="rounded"
                  />
                  <Label htmlFor="weekend">Weekend Transaction</Label>
                </div>

                <div className="flex gap-2">
                  <Button 
                    onClick={handlePrediction} 
                    disabled={isLoading || !transactionData.amount || !transactionData.merchant_category}
                    className="flex-1"
                  >
                    {isLoading ? 'Analyzing...' : 'Check for Fraud'}
                  </Button>
                  <Button variant="outline" onClick={resetForm}>
                    Reset
                  </Button>
                </div>

                {/* Prediction Result */}
                {predictionResult && (
                  <Alert className={predictionResult.prediction === 1 ? 'border-red-500' : 'border-green-500'}>
                    <div className="flex items-center gap-2">
                      {predictionResult.prediction === 1 ? (
                        <AlertTriangle className="h-4 w-4 text-red-500" />
                      ) : (
                        <CheckCircle className="h-4 w-4 text-green-500" />
                      )}
                      <AlertDescription>
                        <strong>
                          {predictionResult.prediction === 1 ? 'FRAUD DETECTED' : 'LEGITIMATE TRANSACTION'}
                        </strong>
                        <br />
                        Confidence: {(predictionResult.confidence * 100).toFixed(1)}%
                      </AlertDescription>
                    </div>
                  </Alert>
                )}
              </CardContent>
            </Card>

            {/* Recent Transactions */}
            <Card className="mt-6">
              <CardHeader>
                <CardTitle>Recent Transactions</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2 max-h-64 overflow-y-auto">
                  {recentTransactions.map(transaction => (
                    <div key={transaction.id} className="flex items-center justify-between p-2 bg-gray-50 rounded">
                      <div className="text-sm">
                        <div className="font-medium">${transaction.amount}</div>
                        <div className="text-gray-500">{transaction.merchant_category}</div>
                      </div>
                      <div className="text-right">
                        <Badge variant={transaction.prediction === 1 ? 'destructive' : 'default'}>
                          {transaction.prediction === 1 ? 'Fraud' : 'Safe'}
                        </Badge>
                        <div className="text-xs text-gray-500">{transaction.timestamp}</div>
                      </div>
                    </div>
                  ))}
                  {recentTransactions.length === 0 && (
                    <p className="text-gray-500 text-center py-4">No recent transactions</p>
                  )}
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Charts and Analytics */}
          <div className="lg:col-span-2 space-y-6">
            {/* Fraud Trends Chart */}
            <Card>
              <CardHeader>
                <CardTitle>Fraud Detection Trends (24h)</CardTitle>
                <CardDescription>Hourly fraud detection patterns</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={fraudTrends}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="time" />
                    <YAxis />
                    <Tooltip />
                    <Line type="monotone" dataKey="fraudCount" stroke="#ef4444" strokeWidth={2} name="Fraud Detected" />
                    <Line type="monotone" dataKey="totalCount" stroke="#3b82f6" strokeWidth={2} name="Total Transactions" />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            {/* Category Distribution */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle>Fraud by Category</CardTitle>
                  <CardDescription>Distribution of fraud across merchant categories</CardDescription>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={250}>
                    <PieChart>
                      <Pie
                        data={categoryData}
                        cx="50%"
                        cy="50%"
                        outerRadius={80}
                        fill="#8884d8"
                        dataKey="value"
                        label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                      >
                        {categoryData.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.color} />
                        ))}
                      </Pie>
                      <Tooltip />
                    </PieChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Model Performance</CardTitle>
                  <CardDescription>Real-time model metrics</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div>
                      <div className="flex justify-between text-sm">
                        <span>Accuracy</span>
                        <span>{systemMetrics.accuracy}%</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div 
                          className="bg-green-600 h-2 rounded-full" 
                          style={{ width: `${systemMetrics.accuracy}%` }}
                        ></div>
                      </div>
                    </div>
                    
                    <div>
                      <div className="flex justify-between text-sm">
                        <span>Precision</span>
                        <span>92.3%</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div className="bg-blue-600 h-2 rounded-full" style={{ width: '92.3%' }}></div>
                      </div>
                    </div>
                    
                    <div>
                      <div className="flex justify-between text-sm">
                        <span>Recall</span>
                        <span>88.7%</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div className="bg-purple-600 h-2 rounded-full" style={{ width: '88.7%' }}></div>
                      </div>
                    </div>
                    
                    <div>
                      <div className="flex justify-between text-sm">
                        <span>F1 Score</span>
                        <span>90.4%</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div className="bg-orange-600 h-2 rounded-full" style={{ width: '90.4%' }}></div>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
