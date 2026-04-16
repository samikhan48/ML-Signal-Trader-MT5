//+------------------------------------------------------------------+
//|                                       ML_Signal_Trader_V5.mq5    |
//|                        Hybrid ML Trading: ONNX + CSV Signals     |
//|                  V5: Institutional Triple-Barrier & Macro Proxy  |
//+------------------------------------------------------------------+
#property copyright "Antigravity AI V5"
#property link      ""
#property version   "5.00"
#property strict

#include <Trade\Trade.mqh>

//=============================================================================
// ENUMERATIONS
//=============================================================================
enum ENUM_SIGNAL_MODE
{
   MODE_CSV_ONLY,       // CSV Only (Python signals)
   MODE_ONNX_ONLY,      // ONNX Only (Internal model)
   MODE_HYBRID_AGREE,   // Hybrid: Both must agree
   MODE_HYBRID_ANY      // Hybrid: Either can trigger
};

//=============================================================================
// INPUT PARAMETERS - Organized by Category
//=============================================================================

//--- Signal Source
input group "🧠 Signal Mode"
input ENUM_SIGNAL_MODE SignalMode = MODE_CSV_ONLY;  // Primary Signal Source
input string   SignalFile = "ml_signals_svm_finbert.csv";  // CSV Signal File
input string   OnnxModelFolder = "ML_ONNX_Models";  // ONNX Model Folder (in Common)
input double   MinConfidence = 0.55;     // Min Confidence Score (0.0-1.0)
input double   OnnxConfThreshold = 0.55; // ONNX Probability Threshold
input string   MacroProxySymbol = "XAUUSD"; // V5 Macro Context Symbol (Gold/VIX)
input ulong    MagicNumber = 202605;     // Magic Number for Orders (V5)

//--- Sentiment Filter
input group "📰 Sentiment Filter (FinBERT)"
input bool     UseSentimentFilter = true;     // Enable Sentiment Filter
input double   SentimentThreshold = 0.30;     // Sentiment Block Level (±0.3)

//--- Regime Filter
input group "🌊 Regime Filter (HMM)"
input bool     UseRegimeFilter = true;        // Enable Regime Filter
input bool     AllowLowVol = true;            // Trade in LOW_VOL Regime
input bool     AllowMediumVol = true;         // Trade in MEDIUM_VOL Regime
input bool     AllowHighVol = false;          // Trade in HIGH_VOL Regime

//--- News Filter
input group "📅 News Filter"
input bool     UseNewsFilter = false;         // Enable News Avoidance
input int      NewsMinutesBefore = 30;        // Minutes Before High Impact News
input int      NewsMinutesAfter = 30;         // Minutes After High Impact News

//--- Trend Filter
input group "📈 Trend Filter (EMA 200)"
input bool     UseTrendFilter = true;         // Enable EMA 200 Trend Filter

//--- Risk Management
input group "⚠️ Risk Management"
input double   RiskPercent = 1.0;             // Risk Per Trade (% of Equity)
input int      ATRPeriod = 14;                // ATR Period for SL/TP
input double   ATRMultSL = 2.0;               // ATR Multiplier for Stop Loss
input double   ATRMultTP = 3.0;               // ATR Multiplier for Take Profit

//--- Trailing Stop
input group "🎯 Trailing Stop"
input bool     UseTrailingStop = true;        // Enable Trailing Stop
input double   TrailActivateATR = 1.5;        // Activate After (ATR units profit)
input double   TrailDistanceATR = 1.0;        // Trail Distance (ATR units)

//--- Daily Loss Limit
input group "🛡️ Daily Protection"
input bool     UseDailyLossLimit = true;      // Enable Daily Loss Limit
input double   DailyLossPercent = 3.0;        // Max Daily Loss (% of equity)

//--- Session Filter
input group "⏰ Session Filter"
input bool     UseSessionFilter = true;       // Enable Session Filter
input int      SessionStartHour = 8;          // Session Start (GMT)
input int      SessionEndHour = 20;           // Session End (GMT)

//=============================================================================
// GLOBAL VARIABLES
//=============================================================================
CTrade         trade;
long           onnx_handle = INVALID_HANDLE;
int            handleATR;
int            handleEMA200;
int            handleADX;

// ONNX outputs
long           output_label[];
float          output_vector[];

// Daily loss tracking
double         dailyStartEquity;
datetime       lastDayCheck;
bool           dailyLimitHit;

// Current model info
string         currentOnnxModel = "";

// Feature count for ONNX (V5 incorporates 20 tech + 1 Macro Proxy)
#define FEATURE_COUNT 21

//+------------------------------------------------------------------+
//| Expert initialization function                                     |
//+------------------------------------------------------------------+
int OnInit()
{
   //--- Initialize ONNX model if needed
   if(SignalMode == MODE_ONNX_ONLY || SignalMode == MODE_HYBRID_AGREE || SignalMode == MODE_HYBRID_ANY)
   {
      if(!LoadOnnxModelForSymbol())
      {
         Print("⚠️ ONNX model not loaded - using CSV mode only");
         if(SignalMode == MODE_ONNX_ONLY)
         {
            Print("❌ ONNX_ONLY mode requires valid ONNX model. Initialization failed.");
            return(INIT_FAILED);
         }
      }
   }
   
   //--- Initialize indicators
   handleATR = iATR(_Symbol, _Period, ATRPeriod);
   if(handleATR == INVALID_HANDLE) return(INIT_FAILED);
   
   handleEMA200 = iMA(_Symbol, _Period, 200, 0, MODE_EMA, PRICE_CLOSE);
   if(handleEMA200 == INVALID_HANDLE) return(INIT_FAILED);
   
   handleADX = iADX(_Symbol, _Period, 14);
   if(handleADX == INVALID_HANDLE) return(INIT_FAILED);
   
   //--- Initialize trade object
   trade.SetExpertMagicNumber(MagicNumber);
   trade.SetDeviationInPoints(10);
   trade.SetTypeFilling(ORDER_FILLING_IOC);
   
   //--- Initialize daily loss tracking
   dailyStartEquity = AccountInfoDouble(ACCOUNT_EQUITY);
   lastDayCheck = 0;
   dailyLimitHit = false;
   
   //--- Print configuration
   Print("═══════════════════════════════════════════════════════════");
   Print("✅ ML Signal Trader V5.00 (Triple-Barrier ONNX & Macro Proxy) Initialized");
   Print("═══════════════════════════════════════════════════════════");
   PrintFormat("📊 Symbol: %s | Timeframe: %s", _Symbol, EnumToString(_Period));
   PrintFormat("🧠 Signal Mode: %s", EnumToString(SignalMode));
   PrintFormat("📁 ONNX Model: %s", currentOnnxModel != "" ? currentOnnxModel : "Not Loaded");
   Print("📊 CSV File: ", SignalFile);
   Print("📊 Min Confidence: ", MinConfidence * 100, "%");
   Print("📰 Sentiment Filter: ", UseSentimentFilter ? "ON" : "OFF");
   Print("🌊 Regime Filter: ", UseRegimeFilter ? "ON" : "OFF");
   Print("📅 News Filter: ", UseNewsFilter ? "ON" : "OFF");
   Print("📈 Trend Filter: ", UseTrendFilter ? "ON" : "OFF");
   Print("🎯 Trailing Stop: ", UseTrailingStop ? "ON" : "OFF");
   Print("🛡️ Daily Limit: ", UseDailyLossLimit ? DoubleToString(DailyLossPercent, 1) + "%" : "OFF");
   Print("═══════════════════════════════════════════════════════════");
   
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Load ONNX model for current symbol                                 |
//+------------------------------------------------------------------+
bool LoadOnnxModelForSymbol()
{
   // Build model path: Common\ML_ONNX_Models\ml_strategy_v4_EURUSD.onnx
   string baseSymbol = _Symbol;
   
   // Remove suffix if present (e.g., EURUSDm -> EURUSD)
   if(StringLen(baseSymbol) > 6)
      baseSymbol = StringSubstr(baseSymbol, 0, 6);
   
   string modelPath = OnnxModelFolder + "\\ml_strategy_v5_" + baseSymbol + ".onnx";
   
   Print("🔍 Looking for ONNX model: ", modelPath);
   
   // Try to load from Common folder
   onnx_handle = OnnxCreate(modelPath, 
                            ONNX_COMMON_FOLDER | ONNX_DEBUG_LOGS);
   
   if(onnx_handle == INVALID_HANDLE)
   {
      // Try alternate naming: ml_strategy_v4_{symbol}.onnx
      modelPath = OnnxModelFolder + "\\ml_strategy_v5_" + _Symbol + ".onnx";
      onnx_handle = OnnxCreate(modelPath, ONNX_COMMON_FOLDER | ONNX_DEBUG_LOGS);
   }
   
   if(onnx_handle == INVALID_HANDLE)
   {
      PrintFormat("❌ Failed to load ONNX model for %s", _Symbol);
      PrintFormat("   Expected path: MQL5\\Files\\Common\\%s", modelPath);
      return false;
   }
   
   // Set input shape
   long input_shape[] = {1, FEATURE_COUNT};
   if(!OnnxSetInputShape(onnx_handle, 0, input_shape))
   {
      Print("❌ Failed to set ONNX input shape");
      OnnxRelease(onnx_handle);
      onnx_handle = INVALID_HANDLE;
      return false;
   }
   
   // Set output shapes
   long output_shape_label[] = {1};
   if(!OnnxSetOutputShape(onnx_handle, 0, output_shape_label))
   {
      OnnxRelease(onnx_handle);
      onnx_handle = INVALID_HANDLE;
      return false;
   }
   
   // LightGBM models output 3 classes even for binary classification
   // Models patched with classlabels_int64s=[0,1,2] to match 3 probability outputs
   long output_shape_probs[] = {1, 3};
   if(!OnnxSetOutputShape(onnx_handle, 1, output_shape_probs))
   {
      PrintFormat("WARNING: OnnxSetOutputShape failed for {1,3}, Error: %d", GetLastError());
   }
   PrintFormat("Output shape set to {1, 3}");
   
   currentOnnxModel = modelPath;
   PrintFormat("✅ ONNX model loaded: %s", modelPath);
   
   return true;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                   |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   if(onnx_handle != INVALID_HANDLE) OnnxRelease(onnx_handle);
   IndicatorRelease(handleATR);
   IndicatorRelease(handleEMA200);
   IndicatorRelease(handleADX);
   Print("ML Signal Trader V4 stopped. Reason: ", reason);
}

//+------------------------------------------------------------------+
//| Expert tick function                                               |
//+------------------------------------------------------------------+
void OnTick()
{
   //--- Always manage trailing stops
   if(UseTrailingStop)
      ManageTrailingStop();
   
   //--- Check for new bar
   if(!IsNewBar())
      return;
   
   //--- Check daily loss limit
   if(UseDailyLossLimit && IsDailyLimitHit())
      return;
   
   //--- Check session filter
   if(UseSessionFilter && !IsWithinSession())
      return;
   
   //--- Get signals based on mode
   int csvSignal = 0, onnxSignal = 0;
   double csvConfidence = 0, onnxConfidence = 0;
   double sentiment = 0;
   string regime = "";
   
   bool csvOk = false, onnxOk = false;
   
   // Get CSV signal if needed
   if(SignalMode == MODE_CSV_ONLY || SignalMode == MODE_HYBRID_AGREE || SignalMode == MODE_HYBRID_ANY)
   {
      string signalTime;
      csvOk = ReadCSVSignal(csvSignal, csvConfidence, signalTime, sentiment, regime);
   }
   
   // Get ONNX signal if needed
   if(SignalMode == MODE_ONNX_ONLY || SignalMode == MODE_HYBRID_AGREE || SignalMode == MODE_HYBRID_ANY)
   {
      if(onnx_handle != INVALID_HANDLE)
         onnxOk = GetOnnxSignal(onnxSignal, onnxConfidence);
   }
   
   //--- Determine final signal based on mode
   int finalSignal = 0;
   double finalConfidence = 0;
   
   switch(SignalMode)
   {
      case MODE_CSV_ONLY:
         if(!csvOk) return;
         finalSignal = csvSignal;
         finalConfidence = csvConfidence;
         break;
         
      case MODE_ONNX_ONLY:
         if(!onnxOk) return;
         finalSignal = onnxSignal;
         finalConfidence = onnxConfidence;
         // No sentiment/regime from ONNX, skip those filters
         sentiment = 0;
         regime = "MEDIUM_VOL";
         break;
         
      case MODE_HYBRID_AGREE:
         if(!csvOk || !onnxOk) return;
         if(csvSignal != onnxSignal)
         {
            PrintFormat("⚠️ Hybrid disagreement: CSV=%d, ONNX=%d", csvSignal, onnxSignal);
            return;
         }
         finalSignal = csvSignal;
         finalConfidence = (csvConfidence + onnxConfidence) / 2.0;
         Print("✅ Hybrid agreement: Both signals = ", finalSignal == 1 ? "BUY" : "SELL");
         break;
         
      case MODE_HYBRID_ANY:
         if(csvOk && csvConfidence >= MinConfidence)
         {
            finalSignal = csvSignal;
            finalConfidence = csvConfidence;
         }
         else if(onnxOk && onnxConfidence >= OnnxConfThreshold)
         {
            finalSignal = onnxSignal;
            finalConfidence = onnxConfidence;
            sentiment = 0;
            regime = "MEDIUM_VOL";
         }
         else
            return;
         break;
   }
   
   //--- Check confidence threshold
   if(finalConfidence < MinConfidence)
   {
      return;
   }
   
   //--- Check regime filter (skip if ONNX-only)
   if(UseRegimeFilter && SignalMode != MODE_ONNX_ONLY && !IsRegimeAllowed(regime))
   {
      PrintFormat("🌊 %s: Blocked by Regime Filter (%s)", _Symbol, regime);
      return;
   }
   
   //--- Check sentiment filter (skip if ONNX-only)
   if(UseSentimentFilter && SignalMode != MODE_ONNX_ONLY && !IsSentimentAllowed(finalSignal, sentiment))
   {
      PrintFormat("📰 %s: Blocked by Sentiment (%.2f contradicts %s)", 
                  _Symbol, sentiment, finalSignal == 1 ? "BUY" : "SELL");
      return;
   }
   
   //--- Check trend filter (EMA 200)
   if(UseTrendFilter && !IsTrendAligned(finalSignal))
   {
      PrintFormat("📈 %s: Blocked by Trend Filter (EMA 200)", _Symbol);
      return;
   }
   
   //--- Check news filter
   if(UseNewsFilter && IsHighImpactNews())
   {
      PrintFormat("📅 %s: Blocked by News Filter", _Symbol);
      return;
   }
   
   //--- Check if already have position
   if(HasPosition())
      return;
   
   //--- Execute trade
   if(finalSignal == 1)
      OpenBuy(finalConfidence);
   else if(finalSignal == -1)
      OpenSell(finalConfidence);
}

//+------------------------------------------------------------------+
//| Read signal from CSV file                                          |
//+------------------------------------------------------------------+
bool ReadCSVSignal(int &signal, double &confidence, string &signalTime, 
                   double &sentiment, string &regime)
{
   int handle = FileOpen(SignalFile, FILE_READ|FILE_CSV|FILE_ANSI|FILE_COMMON, ',');
   
   if(handle == INVALID_HANDLE)
      handle = FileOpen(SignalFile, FILE_READ|FILE_CSV|FILE_ANSI, ',');
   
   if(handle == INVALID_HANDLE)
      return false;
   
   // Skip header
   if(!FileIsEnding(handle))
   {
      while(!FileIsLineEnding(handle) && !FileIsEnding(handle))
         FileReadString(handle);
   }
   
   string currentSymbol = _Symbol;
   
   // Also try base symbol without suffix
   string baseSymbol = _Symbol;
   if(StringLen(baseSymbol) > 6)
      baseSymbol = StringSubstr(baseSymbol, 0, 6);
   
   bool found = false;
   
   while(!FileIsEnding(handle))
   {
      string dateStr = FileReadString(handle);
      string symbol = FileReadString(handle);       // Shifted up!
      string signalStr = FileReadString(handle);
      string confStr = FileReadString(handle);
      string priceStr = FileReadString(handle);
      string regimeStr = FileReadString(handle);
      string sentimentStr = FileReadString(handle);
      string filterStr = FileReadString(handle);    // Added this!
      
      if(symbol == currentSymbol || symbol == baseSymbol)
      {
         signal = (int)StringToInteger(signalStr);
         confidence = StringToDouble(confStr);
         signalTime = dateStr;                      // Fixed this!
         regime = regimeStr;
         sentiment = StringToDouble(sentimentStr);
         found = true;
         break;
      }
   }
   
   FileClose(handle);
   return found;
}

//+------------------------------------------------------------------+
//| Get signal from ONNX model                                         |
//+------------------------------------------------------------------+
bool GetOnnxSignal(int &signal, double &confidence)
{
   if(onnx_handle == INVALID_HANDLE)
      return false;
   
   //--- Get price data
   int count = 250;
   double closes[], highs[], lows[], opens[];
   long volumes[];
   
   ArraySetAsSeries(closes, true);
   ArraySetAsSeries(highs, true);
   ArraySetAsSeries(lows, true);
   ArraySetAsSeries(opens, true);
   ArraySetAsSeries(volumes, true);
   
   if(CopyClose(_Symbol, _Period, 0, count, closes) < count ||
      CopyHigh(_Symbol, _Period, 0, count, highs) < count ||
      CopyLow(_Symbol, _Period, 0, count, lows) < count ||
      CopyOpen(_Symbol, _Period, 0, count, opens) < count ||
      CopyTickVolume(_Symbol, _Period, 0, count, volumes) < count)
      return false;
   
   //--- Get indicator values
   double adx_buf[1], ema200_buf[1];
   if(CopyBuffer(handleADX, 0, 1, 1, adx_buf) < 0) return false;
   if(CopyBuffer(handleEMA200, 0, 1, 1, ema200_buf) < 0) return false;
   
   //--- Get time features
   datetime currentTime = iTime(_Symbol, _Period, 0);
   MqlDateTime dt;
   TimeToStruct(currentTime, dt);
   
   //--- Calculate 21 features (V5 Macro Upgrade)
   float features[FEATURE_COUNT];
   int shift = 1;
   
   // Feature 0: RSI
   features[0] = (float)CalculateRSI(shift, 14, closes);
   
   // Features 1-2: SMA 10, 50
   double sum10=0, sum50=0;
   for(int k=0; k<10; k++) sum10 += closes[shift+k];
   for(int k=0; k<50; k++) sum50 += closes[shift+k];
   features[1] = (float)(sum10/10.0);
   features[2] = (float)(sum50/50.0);
   
   // Feature 3: ADX
   features[3] = (float)adx_buf[0];
   
   // Feature 4: Returns
   features[4] = (float)((closes[shift] - closes[shift+1]) / closes[shift+1]);
   
   // Feature 5: Volatility
   double sumRet=0, sumRetSq=0;
   for(int k=0; k<20; k++)
   {
      double r = (closes[shift+k] - closes[shift+k+1]) / closes[shift+k+1];
      sumRet += r;
      sumRetSq += r*r;
   }
   double meanRet = sumRet/20.0;
   double varRet = sumRetSq/20.0 - meanRet*meanRet;
   if(varRet < 0) varRet = 0;
   features[5] = (float)MathSqrt(varRet);
   
   // Features 6-7: High-Low, Close-Open
   features[6] = (float)(highs[shift] - lows[shift]);
   features[7] = (float)(closes[shift] - opens[shift]);
   
   // Features 8-13: Lags
   for(int k=1; k<=3; k++)
   {
      int s = shift + k;
      features[7+k] = (float)((closes[s] - closes[s+1]) / closes[s+1]);
      features[10+k] = (float)CalculateRSI(s, 14, closes);
   }
   
   // Features 14-15: Time
   features[14] = (float)dt.hour;
   features[15] = (float)dt.day_of_week;
   
   // Feature 16: EMA 200 Position
   features[16] = (closes[shift] > ema200_buf[0]) ? 1.0f : 0.0f;
   
   // Feature 17: ATR Ratio
   double atr14 = GetATR();
   double sumATR = 0;
   for(int k=0; k<50; k++) sumATR += (highs[shift+k] - lows[shift+k]);
   double atr50 = sumATR / 50.0;
   features[17] = (float)((atr50 > 0) ? atr14 / atr50 : 1.0);
   
   // Feature 18: Volume Ratio
   double sumVol = 0;
   for(int k=0; k<20; k++) sumVol += (double)volumes[shift+k];
   double avgVol = sumVol / 20.0;
   features[18] = (float)((avgVol > 0) ? (double)volumes[shift] / avgVol : 1.0);
   
   // Feature 19: Spread Normalized
   features[19] = (float)((atr14 > 0) ? (highs[shift] - lows[shift]) / atr14 : 1.0);
   
   // Feature 20: Macro Context Proxy Z-Score (Phase 7 Injection)
   double proxy_closes[];
   ArraySetAsSeries(proxy_closes, true);
   
   if(CopyClose(MacroProxySymbol, PERIOD_H1, 0, 50, proxy_closes) == 50)
   {
      double proxy_sum = 0;
      for(int k=0; k<50; k++) proxy_sum += proxy_closes[k];
      double mean = proxy_sum / 50.0;
      
      double var_sum = 0;
      for(int k=0; k<50; k++) var_sum += MathPow(proxy_closes[k] - mean, 2);
      double std_dev = MathSqrt(var_sum / 50.0);
      
      features[20] = (float)((std_dev > 0) ? (proxy_closes[0] - mean) / std_dev : 0.0);
   }
   else
   {
      PrintFormat("⚠️ ONNX V5 Warning: Could not fetch %s for Proxy Feature. Defaulting to 0.", MacroProxySymbol);
      features[20] = 0.0f; // Neutral fallback to avoid model crash
   }
   
   // Validate features
   for(int i=0; i<FEATURE_COUNT; i++)
   {
      if(!MathIsValidNumber(features[i]))
         return false;
   }
   
   //--- Run ONNX inference
   ArrayResize(output_label, 1);
   ArrayResize(output_vector, 3);  // LightGBM models output 3 classes
   
   if(!OnnxRun(onnx_handle, ONNX_DEFAULT, features, output_label, output_vector))
   {
      Print("❌ ONNX inference failed - Error: ", GetLastError());
      return false;
   }
   
   // ONNX binary targets natively output the probability of the POSITIVE class (1 = BUY)
   // output_vector[0] holds the exact percentage confidence for a BUY
   double probBuy = output_vector[0];
   double probSell = 1.0 - probBuy; // SELL is mathematically the inverse
   
   PrintFormat("DEBUG ONNX: probSell = %.4f | probBuy = %.4f", probSell, probBuy);
   
   if(probBuy > OnnxConfThreshold && probBuy > probSell)
   {
      signal = 1;  // BUY
      confidence = probBuy;
      return true;
   }
   else if(probSell > OnnxConfThreshold && probSell > probBuy)
   {
      signal = -1;  // SELL
      confidence = probSell;
      return true;
   }
   
   return false;
}

//+------------------------------------------------------------------+
//| Calculate RSI                                                      |
//+------------------------------------------------------------------+
double CalculateRSI(int index, int period, double &close_data[])
{
   double avgGain = 0, avgLoss = 0;
   
   for(int i=0; i<period; i++)
   {
      double change = close_data[index+i] - close_data[index+i+1];
      if(change > 0)
         avgGain += change;
      else
         avgLoss -= change;
   }
   
   avgGain /= period;
   avgLoss = MathAbs(avgLoss) / period;
   
   if(avgLoss == 0) return 100.0;
   double rs = avgGain / avgLoss;
   return 100.0 - (100.0 / (1.0 + rs));
}

//+------------------------------------------------------------------+
//| Check if regime allows trading                                     |
//+------------------------------------------------------------------+
bool IsRegimeAllowed(string regime)
{
   if(regime == "LOW_VOL" && AllowLowVol) return true;
   if(regime == "MEDIUM_VOL" && AllowMediumVol) return true;
   if(regime == "HIGH_VOL" && AllowHighVol) return true;
   return false;
}

//+------------------------------------------------------------------+
//| Check if sentiment allows the trade                                |
//+------------------------------------------------------------------+
bool IsSentimentAllowed(int signal, double sentiment)
{
   if(signal == 1 && sentiment < -SentimentThreshold)
      return false;
   if(signal == -1 && sentiment > SentimentThreshold)
      return false;
   return true;
}

//+------------------------------------------------------------------+
//| Check if trend is aligned (EMA 200)                                |
//+------------------------------------------------------------------+
bool IsTrendAligned(int signal)
{
   double ema200[1];
   if(CopyBuffer(handleEMA200, 0, 1, 1, ema200) < 0) return true;
   
   double close = iClose(_Symbol, _Period, 1);
   
   if(signal == 1 && close < ema200[0])
      return false;
   if(signal == -1 && close > ema200[0])
      return false;
   
   return true;
}

//+------------------------------------------------------------------+
//| Check if within trading session                                    |
//+------------------------------------------------------------------+
bool IsWithinSession()
{
   datetime currentTime = TimeCurrent();
   MqlDateTime dt;
   TimeToStruct(currentTime, dt);
   
   int hour = dt.hour;
   
   if(SessionStartHour < SessionEndHour)
      return (hour >= SessionStartHour && hour < SessionEndHour);
   else
      return (hour >= SessionStartHour || hour < SessionEndHour);
}

//+------------------------------------------------------------------+
//| Get ISO Country Code specifically for MT5 Calendar               |
//+------------------------------------------------------------------+
string GetCountryByCurrency(string currency)
{
   if(currency == "USD") return "US";
   if(currency == "GBP") return "GB";
   if(currency == "EUR") return "EU";
   if(currency == "JPY") return "JP";
   if(currency == "AUD") return "AU";
   if(currency == "NZD") return "NZ";
   if(currency == "CAD") return "CA";
   if(currency == "CHF") return "CH";
   return "";
}

//+------------------------------------------------------------------+
//| Check for high impact news using native MT5 Calendar             |
//+------------------------------------------------------------------+
bool IsHighImpactNews()
{
   // Calculate our specific time window
   datetime currentTime = TimeCurrent();
   datetime timeFrom = currentTime - (NewsMinutesAfter * 60);
   datetime timeTo   = currentTime + (NewsMinutesBefore * 60);
   
   // Extract the two currencies from the current chart symbol (e.g. GBP and USD)
   string baseCurrency  = StringSubstr(_Symbol, 0, 3);
   string quoteCurrency = StringSubstr(_Symbol, 3, 3);
   
   string countryBase  = GetCountryByCurrency(baseCurrency);
   string countryQuote = GetCountryByCurrency(quoteCurrency);
   
   MqlCalendarValue values[];
   
   // 1. Check the Base Currency for High Impact Events
   if(countryBase != "" && CalendarValueHistory(values, timeFrom, timeTo, countryBase))
   {
      for(int i = 0; i < ArraySize(values); i++)
      {
         MqlCalendarEvent event;
         if(CalendarEventById(values[i].event_id, event))
         {
            if(event.importance == CALENDAR_IMPORTANCE_HIGH)
            {
               PrintFormat("🛑 NEWS PAUSE: Detected High Impact Event: %s (%s)", event.name, countryBase);
               return true; // Stop Trading
            }
         }
      }
   }
   
   // 2. Check the Quote Currency for High Impact Events
   if(countryQuote != "" && CalendarValueHistory(values, timeFrom, timeTo, countryQuote))
   {
      for(int i = 0; i < ArraySize(values); i++)
      {
         MqlCalendarEvent event;
         if(CalendarEventById(values[i].event_id, event))
         {
            if(event.importance == CALENDAR_IMPORTANCE_HIGH)
            {
               PrintFormat("🛑 NEWS PAUSE: Detected High Impact Event: %s (%s)", event.name, countryQuote);
               return true; // Stop Trading
            }
         }
      }
   }
   
   return false; // Coast is clear, no high-impact news in our window
}

//+------------------------------------------------------------------+
//| Check daily loss limit                                             |
//+------------------------------------------------------------------+
bool IsDailyLimitHit()
{
   datetime currentTime = TimeCurrent();
   MqlDateTime dt;
   TimeToStruct(currentTime, dt);
   
   datetime today = StringToTime(IntegerToString(dt.year) + "." + 
                                  IntegerToString(dt.mon) + "." + 
                                  IntegerToString(dt.day));
   
   if(today != lastDayCheck)
   {
      lastDayCheck = today;
      dailyStartEquity = AccountInfoDouble(ACCOUNT_EQUITY);
      dailyLimitHit = false;
   }
   
   if(dailyLimitHit) return true;
   
   double currentEquity = AccountInfoDouble(ACCOUNT_EQUITY);
   double lossPercent = (dailyStartEquity - currentEquity) / dailyStartEquity * 100.0;
   
   if(lossPercent >= DailyLossPercent)
   {
      dailyLimitHit = true;
      Print("🛡️ DAILY LOSS LIMIT HIT! No more trades today.");
      return true;
   }
   
   return false;
}

//+------------------------------------------------------------------+
//| Check if we already have a position                                |
//+------------------------------------------------------------------+
bool HasPosition()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(!PositionSelectByTicket(PositionGetTicket(i))) continue;
      if(PositionGetInteger(POSITION_MAGIC) != MagicNumber) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      return true;
   }
   return false;
}

//+------------------------------------------------------------------+
//| Check for new bar                                                  |
//+------------------------------------------------------------------+
bool IsNewBar()
{
   static datetime lastBar = 0;
   datetime currentBar = iTime(_Symbol, _Period, 0);
   
   if(lastBar != currentBar)
   {
      lastBar = currentBar;
      return true;
   }
   return false;
}

//+------------------------------------------------------------------+
//| Get ATR value                                                      |
//+------------------------------------------------------------------+
double GetATR()
{
   double buf[1];
   if(CopyBuffer(handleATR, 0, 1, 1, buf) > 0)
      return buf[0];
   return 0.0010;
}

//+------------------------------------------------------------------+
//| Calculate position size based on risk %                            |
//+------------------------------------------------------------------+
double CalculateLotSize(double slPoints)
{
   double riskMoney = AccountInfoDouble(ACCOUNT_EQUITY) * RiskPercent / 100.0;
   
   double tickValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double tickSize = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   
   if(tickSize == 0 || slPoints == 0) return 0.01;
   
   double slValue = slPoints * (tickValue / tickSize);
   double lot = riskMoney / slValue;
   
   lot = NormalizeDouble(lot, 2);
   
   double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double maxLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   
   if(lot < minLot) lot = minLot;
   if(lot > maxLot) lot = maxLot;
   
   return lot;
}

//+------------------------------------------------------------------+
//| Open BUY order                                                     |
//+------------------------------------------------------------------+
void OpenBuy(double confidence)
{
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double atr = GetATR();
   
   double sl = ask - atr * ATRMultSL;
   double tp = ask + atr * ATRMultTP;
   double slPoints = atr * ATRMultSL;
   double lot = CalculateLotSize(slPoints);
   
   sl = NormalizeDouble(sl, _Digits);
   tp = NormalizeDouble(tp, _Digits);
   
   string comment = StringFormat("ML V4 BUY [%.1f%%]", confidence * 100);
   
   if(trade.Buy(lot, _Symbol, ask, sl, tp, comment))
   {
      PrintFormat("📈 BUY %s @ %.5f | Lot: %.2f | SL: %.5f | TP: %.5f | Conf: %.1f%%",
                  _Symbol, ask, lot, sl, tp, confidence * 100);
   }
   else
      Print("❌ Buy failed - Error: ", GetLastError());
}

//+------------------------------------------------------------------+
//| Open SELL order                                                    |
//+------------------------------------------------------------------+
void OpenSell(double confidence)
{
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double atr = GetATR();
   
   double sl = bid + atr * ATRMultSL;
   double tp = bid - atr * ATRMultTP;
   double slPoints = atr * ATRMultSL;
   double lot = CalculateLotSize(slPoints);
   
   sl = NormalizeDouble(sl, _Digits);
   tp = NormalizeDouble(tp, _Digits);
   
   string comment = StringFormat("ML V4 SELL [%.1f%%]", confidence * 100);
   
   if(trade.Sell(lot, _Symbol, bid, sl, tp, comment))
   {
      PrintFormat("📉 SELL %s @ %.5f | Lot: %.2f | SL: %.5f | TP: %.5f | Conf: %.1f%%",
                  _Symbol, bid, lot, sl, tp, confidence * 100);
   }
   else
      Print("❌ Sell failed - Error: ", GetLastError());
}

//+------------------------------------------------------------------+
//| Manage trailing stop for open positions                            |
//+------------------------------------------------------------------+
void ManageTrailingStop()
{
   double atr = GetATR();
   if(atr == 0) return;
   
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(!PositionSelectByTicket(PositionGetTicket(i))) continue;
      if(PositionGetInteger(POSITION_MAGIC) != MagicNumber) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      
      double openPrice = PositionGetDouble(POSITION_PRICE_OPEN);
      double currentSL = PositionGetDouble(POSITION_SL);
      double tp = PositionGetDouble(POSITION_TP);
      long posType = PositionGetInteger(POSITION_TYPE);
      ulong ticket = PositionGetInteger(POSITION_TICKET);
      
      double trailActivate = TrailActivateATR * atr;
      double trailDistance = TrailDistanceATR * atr;
      
      if(posType == POSITION_TYPE_BUY)
      {
         double currentPrice = SymbolInfoDouble(_Symbol, SYMBOL_BID);
         double profit = currentPrice - openPrice;
         
         if(profit >= trailActivate)
         {
            double newSL = currentPrice - trailDistance;
            newSL = NormalizeDouble(newSL, _Digits);
            
            if(newSL > currentSL + _Point)
               trade.PositionModify(ticket, newSL, tp);
         }
      }
      else if(posType == POSITION_TYPE_SELL)
      {
         double currentPrice = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
         double profit = openPrice - currentPrice;
         
         if(profit >= trailActivate)
         {
            double newSL = currentPrice + trailDistance;
            newSL = NormalizeDouble(newSL, _Digits);
            
            if(newSL < currentSL - _Point || currentSL == 0)
               trade.PositionModify(ticket, newSL, tp);
         }
      }
   }
}
//+------------------------------------------------------------------+
