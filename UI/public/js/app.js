/**
 * NETRA v2 — Suno Sutra Device UI
 * Fixed 320×240 TFT interface: Voice + Camera + Data Inject
 * Offline-first, multilingual AI pipeline
 */

const API = window.location.origin;
let currentUILang = localStorage.getItem("netra_lang") || "";
let isRecording = false;

const LANG_NAMES = {
  hi: "हिन्दी", en: "English", ta: "தமிழ்", te: "తెలుగు", bn: "বাংলা",
  mr: "मराठी", gu: "ગુજરાતી", kn: "ಕನ್ನಡ", ml: "മലയാളം", pa: "ਪੰਜਾਬੀ",
  or: "ଓଡ଼ିଆ", as: "অসমীয়া", ur: "اردو", sa: "संस्कृत", kok: "कोंकणी",
  ks: "کٲشُر", mai: "मैथिली", mni: "মৈতৈলোন্", ne: "नेपाली", sd: "سنڌي",
  doi: "डोगरी", brx: "बड़ो", sat: "ᱥᱟᱱᱛᱟᱲᱤ",
};
let mediaRecorder = null;
let audioChunks = [];
let recordingStartTime = 0;
let recordingMimeType = "audio/webm";
let cameraStream = null;
let hasAudioResult = false;
let detectedSpokenLang = null;

// ── Translations ─────────────────────────────────────
const T = {
  hi: {
    greeting: "🙏 नमस्ते",
    idleHint: "बोलो, फोटो लो या अपलोड करो",
    voiceLabel: "बोलो",
    cameraLabel: "देखो",
    uploadLabel: "अपलोड",
    injectBtnLabel: "डेटा",
    questionLabel: "प्रश्न",
    uploadBadType: "कृपया छवि या PDF अपलोड करें।",
    listeningLabel: "सुन रहा है...",
    processingLabel: "समझ रहा है...",
    verifyTitle: "आपने कहा:",
    injectLabel: "ज्ञान भंडार",
    injectFileText: "फ़ाइल चुनो",
    injectSuccess: "✅ डेटा जोड़ दिया गया!",
    injectFail: "❌ कुछ गड़बड़ हुई",
    injectProcessing: "⏳ डेटा पढ़ रहा है...",
    resultPlaying: "🔊 सुनाई दे रहा है...",
    copied: "✅ कॉपी हुआ!",
    injectBulkText: "कई फ़ाइलें",
    scanInboxText: "इनबॉक्स स्कैन",
    kbDocsLabel: "दस्तावेज़",
    kbChunksLabel: "भाग",
    watcherLabel: "स्वतः",
    inboxHint: "📂 Backend/data/inbox/ में फ़ाइलें डालो — अपने आप जुड़ जाएगा",
    bulkUploading: "फ़ाइलें अपलोड हो रही हैं...",
    bulkResult: "सफल: {s}, असफल: {f}",
    filesInInbox: "फ़ाइलें इनबॉक्स में",
    stageNames: { asr: "🗣 सुनना", nmt: "🔄 अनुवाद", llm: "🧠 सोचना", tts: "🔊 बोलना" },
    sosLabel: "आपातकाल",
    weatherLabel: "मौसम",
    mandiLabel: "मंडी",
    remindersLabel: "याद",
    locationTitle: "अपना स्थान चुनें",
    pincodePlaceholder: "पिन कोड दर्ज करें (जैसे 226001)",
    schemesTitle: "आपके क्षेत्र की योजनाएँ:",
    sosHeader: "🚨 आपातकाल",
    weatherHeader: "🌤 मौसम चेतावनी",
    mandiHeader: "🌾 मंडी भाव",
    remindersHeader: "📅 योजना अंतिम तिथि",
    setLocationFirst: "📍 पहले स्थान चुनें",
    clickPincode: "📍 पिन कोड सेट करें",
    loading: "लोड हो रहा...",
    noAlerts: "कोई चेतावनी नहीं",
    offline: "ऑफ़लाइन",
    noData: "कोई डेटा नहीं",
    noReminders: "कोई याद नहीं",
  },
  en: {
    greeting: "🙏 Hello",
    idleHint: "Speak, photo, or upload",
    voiceLabel: "Voice",
    cameraLabel: "See",
    uploadLabel: "Upload",
    injectBtnLabel: "Data",
    questionLabel: "Q",
    uploadBadType: "Please upload an image or PDF.",
    listeningLabel: "Listening...",
    processingLabel: "Thinking...",
    verifyTitle: "You said:",
    injectLabel: "Knowledge Base",
    injectFileText: "Choose file",
    injectSuccess: "✅ Data added!",
    injectFail: "❌ Something went wrong",
    injectProcessing: "⏳ Processing data...",
    resultPlaying: "🔊 Playing audio...",
    copied: "✅ Copied!",
    injectBulkText: "Multiple files",
    scanInboxText: "Scan Inbox",
    kbDocsLabel: "Documents",
    kbChunksLabel: "Chunks",
    watcherLabel: "Auto",
    inboxHint: "📂 Drop files in Backend/data/inbox/ — auto-ingested",
    bulkUploading: "Uploading files...",
    bulkResult: "Success: {s}, Failed: {f}",
    filesInInbox: "files in inbox",
    stageNames: { asr: "🗣 Listen", nmt: "🔄 Translate", llm: "🧠 Think", tts: "🔊 Speak" },
    sosLabel: "SOS",
    weatherLabel: "Weather",
    mandiLabel: "Mandi",
    remindersLabel: "Reminders",
    locationTitle: "Set Your Location",
    pincodePlaceholder: "Enter PIN Code (e.g. 226001)",
    schemesTitle: "Schemes for your area:",
    sosHeader: "🚨 Emergency",
    weatherHeader: "🌤 Weather Alerts",
    mandiHeader: "🌾 Mandi Prices",
    remindersHeader: "📅 Scheme Deadlines",
    setLocationFirst: "📍 Set location first",
    clickPincode: "📍 Click to set PIN code",
    loading: "Loading...",
    noAlerts: "No alerts",
    offline: "Offline",
    noData: "No data",
    noReminders: "No reminders",
  },
  ta: {
    greeting: "🙏 வணக்கம்",
    idleHint: "பேசுங்கள் அல்லது படம் எடுங்கள்",
    voiceLabel: "பேசு",
    cameraLabel: "பார்",
    uploadLabel: "பதிவேற்று",
    injectBtnLabel: "தரவு",
    listeningLabel: "கேட்கிறது...",
    processingLabel: "புரிந்துகொள்கிறது...",
    injectLabel: "அறிவுத் தளம்",
    injectFileText: "கோப்பு தேர்வு",
    injectSuccess: "✅ தரவு சேர்க்கப்பட்டது!",
    injectFail: "❌ பிழை ஏற்பட்டது",
    injectProcessing: "⏳ தரவை படிக்கிறது...",
    resultPlaying: "🔊 ஒலிக்கிறது...",
    copied: "✅ நகலெடுக்கப்பட்டது!",
    stageNames: { asr: "🗣 கேட்டல்", nmt: "🔄 மொழிபெயர்ப்பு", llm: "🧠 சிந்தனை", tts: "🔊 பேச்சு" },
    sosLabel: "அவசரம்",
    weatherLabel: "வானிலை",
    mandiLabel: "மண்டி",
    remindersLabel: "நினைவூட்டல்",
    locationTitle: "உங்கள் இடத்தை தேர்வு செய்க",
    pincodePlaceholder: "பின் குறியீடு உள்ளிடவும் (எ.கா. 226001)",
    schemesTitle: "உங்கள் பகுதி திட்டங்கள்:",
    sosHeader: "🚨 அவசரம்",
    weatherHeader: "🌤 வானிலை எச்சரிக்கை",
    mandiHeader: "🌾 மண்டி விலை",
    remindersHeader: "📅 திட்ட காலக்கெடு",
    setLocationFirst: "📍 முதலில் இடம் தேர்வு செய்க",
    clickPincode: "📍 பின் குறியீடு அமைக்கவும்",
    loading: "ஏற்றுகிறது...",
    noAlerts: "எச்சரிக்கை இல்லை",
    offline: "ஆஃப்லைன்",
    noData: "தரவு இல்லை",
    noReminders: "நினைவூட்டல் இல்லை",
  },
  te: {
    greeting: "🙏 నమస్కారం",
    idleHint: "మాట్లాడండి లేదా ఫోటో తీయండి",
    voiceLabel: "చెప్పు",
    cameraLabel: "చూడు",
    uploadLabel: "అప్‌లోడ్",
    injectBtnLabel: "డేటా",
    listeningLabel: "వింటోంది...",
    processingLabel: "అర్థం చేసుకుంటోంది...",
    injectLabel: "జ్ఞాన భండారం",
    injectFileText: "ఫైల్ ఎంచుకో",
    injectSuccess: "✅ డేటా చేర్చబడింది!",
    injectFail: "❌ ఏదో తప్పు జరిగింది",
    injectProcessing: "⏳ డేటా చదువుతోంది...",
    resultPlaying: "🔊 వినిపిస్తోంది...",
    copied: "✅ కాపీ అయింది!",
    stageNames: { asr: "🗣 వినడం", nmt: "🔄 అనువాదం", llm: "🧠 ఆలోచన", tts: "🔊 చెప్పడం" },
    sosLabel: "అత్యవసరం",
    weatherLabel: "వాతావరణం",
    mandiLabel: "మండి",
    remindersLabel: "గుర్తు",
    locationTitle: "మీ స్థానాన్ని ఎంచుకోండి",
    pincodePlaceholder: "పిన్ కోడ్ నమోదు చేయండి (ఉదా. 226001)",
    schemesTitle: "మీ ప్రాంత పథకాలు:",
    sosHeader: "🚨 అత్యవసరం",
    weatherHeader: "🌤 వాతావరణ హెచ్చరిక",
    mandiHeader: "🌾 మండి ధరలు",
    remindersHeader: "📅 పథక గడువు",
    setLocationFirst: "📍 ముందు స్థానం ఎంచుకోండి",
    clickPincode: "📍 పిన్ కోడ్ సెట్ చేయండి",
    loading: "లోడ్ అవుతోంది...",
    noAlerts: "హెచ్చరికలు లేవు",
    offline: "ఆఫ్‌లైన్",
    noData: "డేటా లేదు",
    noReminders: "గుర్తులు లేవు",
  },
  bn: {
    greeting: "🙏 নমস্কার",
    idleHint: "বলো বা ছবি তোলো",
    voiceLabel: "বলো",
    cameraLabel: "দেখো",
    uploadLabel: "আপলোড",
    injectBtnLabel: "ডেটা",
    listeningLabel: "শুনছে...",
    processingLabel: "বুঝছে...",
    injectLabel: "জ্ঞান ভান্ডার",
    injectFileText: "ফাইল বেছে নাও",
    injectSuccess: "✅ ডেটা যোগ হয়েছে!",
    injectFail: "❌ কিছু সমস্যা হয়েছে",
    injectProcessing: "⏳ ডেটা পড়ছে...",
    resultPlaying: "🔊 শোনা যাচ্ছে...",
    copied: "✅ কপি হয়েছে!",
    stageNames: { asr: "🗣 শোনা", nmt: "🔄 অনুবাদ", llm: "🧠 চিন্তা", tts: "🔊 বলা" },
    sosLabel: "জরুরি",
    weatherLabel: "আবহাওয়া",
    mandiLabel: "মান্ডি",
    remindersLabel: "মনে রাখা",
    locationTitle: "আপনার অবস্থান নির্বাচন করুন",
    pincodePlaceholder: "পিন কোড লিখুন (যেমন 226001)",
    schemesTitle: "আপনার এলাকার প্রকল্প:",
    sosHeader: "🚨 জরুরি",
    weatherHeader: "🌤 আবহাওয়া সতর্কতা",
    mandiHeader: "🌾 মান্ডি দাম",
    remindersHeader: "📅 প্রকল্পের শেষ তারিখ",
    setLocationFirst: "📍 আগে অবস্থান নির্বাচন করুন",
    clickPincode: "📍 পিন কোড সেট করুন",
    loading: "লোড হচ্ছে...",
    noAlerts: "কোনো সতর্কতা নেই",
    offline: "অফলাইন",
    noData: "কোনো তথ্য নেই",
    noReminders: "কোনো স্মারক নেই",
  },
  mr: {
    greeting: "🙏 नमस्कार",
    idleHint: "बोला किंवा फोटो घ्या",
    voiceLabel: "बोला",
    cameraLabel: "बघा",
    uploadLabel: "अपलोड",
    injectBtnLabel: "डेटा",
    listeningLabel: "ऐकत आहे...",
    processingLabel: "समजत आहे...",
    injectLabel: "ज्ञान भांडार",
    injectFileText: "फाइल निवडा",
    injectSuccess: "✅ डेटा जोडला!",
    injectFail: "❌ काही चूक झाली",
    injectProcessing: "⏳ डेटा वाचत आहे...",
    resultPlaying: "🔊 ऐकवत आहे...",
    copied: "✅ कॉपी झाले!",
    stageNames: { asr: "🗣 ऐकणे", nmt: "🔄 भाषांतर", llm: "🧠 विचार", tts: "🔊 बोलणे" },
    sosLabel: "आणीबाणी",
    weatherLabel: "हवामान",
    mandiLabel: "बाजार",
    remindersLabel: "स्मरण",
    locationTitle: "तुमचे स्थान निवडा",
    pincodePlaceholder: "पिन कोड टाका (उदा. 226001)",
    schemesTitle: "तुमच्या भागातील योजना:",
    sosHeader: "🚨 आणीबाणी",
    weatherHeader: "🌤 हवामान इशारा",
    mandiHeader: "🌾 बाजार भाव",
    remindersHeader: "📅 योजना अंतिम तारीख",
    setLocationFirst: "📍 आधी स्थान निवडा",
    clickPincode: "📍 पिन कोड सेट करा",
    loading: "लोड होत आहे...",
    noAlerts: "इशारे नाहीत",
    offline: "ऑफलाइन",
    noData: "डेटा नाही",
    noReminders: "स्मरणे नाहीत",
  },
  gu: {
    greeting: "🙏 નમસ્તે",
    idleHint: "બોલો અથવા ફોટો લો",
    voiceLabel: "બોલો",
    cameraLabel: "જુઓ",
    uploadLabel: "અપલોડ",
    injectBtnLabel: "ડેટા",
    listeningLabel: "સાંભળે છે...",
    processingLabel: "સમજે છે...",
    injectLabel: "જ્ઞાન ભંડાર",
    injectFileText: "ફાઇલ પસંદ કરો",
    injectSuccess: "✅ ડેટા ઉમેરાયો!",
    injectFail: "❌ કંઈક ખોટું થયું",
    injectProcessing: "⏳ ડેટા વાંચે છે...",
    resultPlaying: "🔊 સંભળાઈ રહ્યું છે...",
    copied: "✅ કૉપી થયું!",
    stageNames: { asr: "🗣 સાંભળવું", nmt: "🔄 અનુવાદ", llm: "🧠 વિચાર", tts: "🔊 બોલવું" },
    sosLabel: "કટોકટી",
    weatherLabel: "હવામાન",
    mandiLabel: "મંડી",
    remindersLabel: "યાદ",
    locationTitle: "તમારું સ્થાન પસંદ કરો",
    pincodePlaceholder: "પિન કોડ દાખલ કરો (દા.ત. 226001)",
    schemesTitle: "તમારા વિસ્તારની યોજનાઓ:",
    sosHeader: "🚨 કટોકટી",
    weatherHeader: "🌤 હવામાન ચેતવણી",
    mandiHeader: "🌾 મંડી ભાવ",
    remindersHeader: "📅 યોજના અંતિમ તારીખ",
    setLocationFirst: "📍 પહેલાં સ્થાન પસંદ કરો",
    clickPincode: "📍 પિન કોડ સેટ કરો",
    loading: "લોડ થઈ રહ્યું છે...",
    noAlerts: "કોઈ ચેતવણી નથી",
    offline: "ઑફલાઇન",
    noData: "કોઈ ડેટા નથી",
    noReminders: "કોઈ યાદ નથી",
  },
  kn: {
    greeting: "🙏 ನಮಸ್ಕಾರ",
    idleHint: "ಮಾತಾಡಿ ಅಥವಾ ಫೋಟೋ ತೆಗೆಯಿರಿ",
    voiceLabel: "ಹೇಳಿ",
    cameraLabel: "ನೋಡಿ",
    uploadLabel: "ಅಪ್‌ಲೋಡ್",
    injectBtnLabel: "ಡೇಟಾ",
    listeningLabel: "ಕೇಳುತ್ತಿದೆ...",
    processingLabel: "ಅರ್ಥಮಾಡಿಕೊಳ್ಳುತ್ತಿದೆ...",
    injectLabel: "ಜ್ಞಾನ ಭಂಡಾರ",
    injectFileText: "ಫೈಲ್ ಆಯ್ಕೆ",
    injectSuccess: "✅ ಡೇಟಾ ಸೇರಿಸಲಾಗಿದೆ!",
    injectFail: "❌ ಏನೋ ತಪ್ಪಾಗಿದೆ",
    injectProcessing: "⏳ ಡೇಟಾ ಓದುತ್ತಿದೆ...",
    resultPlaying: "🔊 ಕೇಳಿಸುತ್ತಿದೆ...",
    copied: "✅ ನಕಲಾಗಿದೆ!",
    stageNames: { asr: "🗣 ಕೇಳುವಿಕೆ", nmt: "🔄 ಅನುವಾದ", llm: "🧠 ಆಲೋಚನೆ", tts: "🔊 ಮಾತು" },
    sosLabel: "ತುರ್ತು",
    weatherLabel: "ಹವಾಮಾನ",
    mandiLabel: "ಮಂಡಿ",
    remindersLabel: "ನೆನಪು",
    locationTitle: "ನಿಮ್ಮ ಸ್ಥಳ ಆಯ್ಕೆಮಾಡಿ",
    pincodePlaceholder: "ಪಿನ್ ಕೋಡ್ ನಮೂದಿಸಿ (ಉದಾ. 226001)",
    schemesTitle: "ನಿಮ್ಮ ಪ್ರದೇಶದ ಯೋಜನೆಗಳು:",
    sosHeader: "🚨 ತುರ್ತು",
    weatherHeader: "🌤 ಹವಾಮಾನ ಎಚ್ಚರಿಕೆ",
    mandiHeader: "🌾 ಮಂಡಿ ಬೆಲೆ",
    remindersHeader: "📅 ಯೋಜನೆ ಗಡುವು",
    setLocationFirst: "📍 ಮೊದಲು ಸ್ಥಳ ಆಯ್ಕೆಮಾಡಿ",
    clickPincode: "📍 ಪಿನ್ ಕೋಡ್ ಸೆಟ್ ಮಾಡಿ",
    loading: "ಲೋಡ್ ಆಗುತ್ತಿದೆ...",
    noAlerts: "ಎಚ್ಚರಿಕೆಗಳಿಲ್ಲ",
    offline: "ಆಫ್‌ಲೈನ್",
    noData: "ಡೇಟಾ ಇಲ್ಲ",
    noReminders: "ನೆನಪುಗಳಿಲ್ಲ",
  },
  ml: {
    greeting: "🙏 നമസ്കാരം",
    idleHint: "സംസാരിക്കൂ അല്ലെങ്കില്‍ ഫോട്ടോ എടുക്കൂ",
    voiceLabel: "പറയൂ",
    cameraLabel: "കാണൂ",
    uploadLabel: "അപ്‌ലോഡ്",
    injectBtnLabel: "ഡേറ്റ",
    listeningLabel: "കേള്‍ക്കുന്നു...",
    processingLabel: "മനസ്സിലാക്കുന്നു...",
    injectLabel: "വിജ്ഞാന ശേഖരം",
    injectFileText: "ഫയല്‍ തിരഞ്ഞെടുക്കൂ",
    injectSuccess: "✅ ഡേറ്റ ചേര്‍ത്തു!",
    injectFail: "❌ എന്തോ പിശക് സംഭവിച്ചു",
    injectProcessing: "⏳ ഡേറ്റ വായിക്കുന്നു...",
    resultPlaying: "🔊 കേള്‍ക്കുന്നു...",
    copied: "✅ കോപ്പി ആയി!",
    stageNames: { asr: "🗣 കേള്‍വി", nmt: "🔄 വിവര്‍ത്തനം", llm: "🧠 ചിന്ത", tts: "🔊 സംസാരം" },
    sosLabel: "അടിയന്തരം",
    weatherLabel: "കാലാവസ്ഥ",
    mandiLabel: "മണ്ഡി",
    remindersLabel: "ഓര്‍മ",
    locationTitle: "നിങ്ങളുടെ സ്ഥലം തിരഞ്ഞെടുക്കൂ",
    pincodePlaceholder: "പിന്‍ കോഡ് നല്‍കൂ (ഉദാ. 226001)",
    schemesTitle: "നിങ്ങളുടെ പ്രദേശത്തെ പദ്ധതികള്‍:",
    sosHeader: "🚨 അടിയന്തരം",
    weatherHeader: "🌤 കാലാവസ്ഥ മുന്നറിയിപ്പ്",
    mandiHeader: "🌾 മണ്ഡി വില",
    remindersHeader: "📅 പദ്ധതി അവസാന തീയതി",
    setLocationFirst: "📍 ആദ്യം സ്ഥലം തിരഞ്ഞെടുക്കൂ",
    clickPincode: "📍 പിന്‍ കോഡ് സെറ്റ് ചെയ്യൂ",
    loading: "ലോഡ് ചെയ്യുന്നു...",
    noAlerts: "മുന്നറിയിപ്പുകളില്ല",
    offline: "ഓഫ്‌ലൈന്‍",
    noData: "ഡേറ്റ ഇല്ല",
    noReminders: "ഓര്‍മകളില്ല",
  },
  pa: {
    greeting: "🙏 ਸਤ ਸ੍ਰੀ ਅਕਾਲ",
    idleHint: "ਬੋਲੋ ਜਾਂ ਫੋਟੋ ਖਿੱਚੋ",
    voiceLabel: "ਬੋਲੋ",
    cameraLabel: "ਵੇਖੋ",
    uploadLabel: "ਅਪਲੋਡ",
    injectBtnLabel: "ਡੇਟਾ",
    listeningLabel: "ਸੁਣ ਰਿਹਾ...",
    processingLabel: "ਸਮਝ ਰਿਹਾ...",
    injectLabel: "ਗਿਆਨ ਭੰਡਾਰ",
    injectFileText: "ਫਾਈਲ ਚੁਣੋ",
    injectSuccess: "✅ ਡੇਟਾ ਜੋੜਿਆ!",
    injectFail: "❌ ਕੁਝ ਗਲਤ ਹੋਇਆ",
    injectProcessing: "⏳ ਡੇਟਾ ਪੜ੍ਹ ਰਿਹਾ...",
    resultPlaying: "🔊 ਸੁਣਾਈ ਦੇ ਰਿਹਾ...",
    copied: "✅ ਕਾਪੀ ਹੋ ਗਈ!",
    stageNames: { asr: "🗣 ਸੁਣਨਾ", nmt: "🔄 ਅਨੁਵਾਦ", llm: "🧠 ਸੋਚ", tts: "🔊 ਬੋਲਣਾ" },
    sosLabel: "ਐਮਰਜੈਂਸੀ",
    weatherLabel: "ਮੌਸਮ",
    mandiLabel: "ਮੰਡੀ",
    remindersLabel: "ਯਾਦ",
    locationTitle: "ਆਪਣਾ ਸਥਾਨ ਚੁਣੋ",
    pincodePlaceholder: "ਪਿੰਨ ਕੋਡ ਦਰਜ ਕਰੋ (ਜਿਵੇਂ 226001)",
    schemesTitle: "ਤੁਹਾਡੇ ਖੇਤਰ ਦੀਆਂ ਯੋਜਨਾਵਾਂ:",
    sosHeader: "🚨 ਐਮਰਜੈਂਸੀ",
    weatherHeader: "🌤 ਮੌਸਮ ਚੇਤਾਵਨੀ",
    mandiHeader: "🌾 ਮੰਡੀ ਭਾਅ",
    remindersHeader: "📅 ਯੋਜਨਾ ਆਖ਼ਰੀ ਤਾਰੀਖ਼",
    setLocationFirst: "📍 ਪਹਿਲਾਂ ਸਥਾਨ ਚੁਣੋ",
    clickPincode: "📍 ਪਿੰਨ ਕੋਡ ਸੈੱਟ ਕਰੋ",
    loading: "ਲੋਡ ਹੋ ਰਿਹਾ...",
    noAlerts: "ਕੋਈ ਚੇਤਾਵਨੀ ਨਹੀਂ",
    offline: "ਆਫ਼ਲਾਈਨ",
    noData: "ਕੋਈ ਡੇਟਾ ਨਹੀਂ",
    noReminders: "ਕੋਈ ਯਾਦ ਨਹੀਂ",
  },
  or: {
    greeting: "🙏 ନମସ୍କାର",
    idleHint: "କୁହ ବା ଫଟୋ ନିଅ",
    voiceLabel: "କୁହ",
    cameraLabel: "ଦେଖ",
    uploadLabel: "ଅପଲୋଡ୍",
    injectBtnLabel: "ଡାଟା",
    listeningLabel: "ଶୁଣୁଛି...",
    processingLabel: "ବୁଝୁଛି...",
    injectLabel: "ଜ୍ଞାନ ଭଣ୍ଡାର",
    injectFileText: "ଫାଇଲ୍ ବାଛ",
    injectSuccess: "✅ ଡାଟା ଯୋଡ଼ାଗଲା!",
    injectFail: "❌ କିଛି ଭୁଲ ହେଲା",
    injectProcessing: "⏳ ଡାଟା ପଢ଼ୁଛି...",
    resultPlaying: "🔊 ଶୁଣାଯାଉଛି...",
    copied: "✅ କପି ହେଲା!",
    stageNames: { asr: "🗣 ଶୁଣିବା", nmt: "🔄 ଅନୁବାଦ", llm: "🧠 ଚିନ୍ତା", tts: "🔊 କହିବା" },
    sosLabel: "ଜରୁରୀ",
    weatherLabel: "ପାଣିପାଗ",
    mandiLabel: "ମଣ୍ଡି",
    remindersLabel: "ସ୍ମାରକ",
    locationTitle: "ଆପଣଙ୍କ ସ୍ଥାନ ବାଛନ୍ତୁ",
    pincodePlaceholder: "ପିନ୍ କୋଡ୍ ଲେଖନ୍ତୁ (ଯଥା 226001)",
    schemesTitle: "ଆପଣଙ୍କ ଅଞ୍ଚଳର ଯୋଜନା:",
    sosHeader: "🚨 ଜରୁରୀ",
    weatherHeader: "🌤 ପାଣିପାଗ ସତର୍କତା",
    mandiHeader: "🌾 ମଣ୍ଡି ଦାମ",
    remindersHeader: "📅 ଯୋଜନା ଶେଷ ତାରିଖ",
    setLocationFirst: "📍 ପ୍ରଥମେ ସ୍ଥାନ ବାଛନ୍ତୁ",
    clickPincode: "📍 ପିନ୍ କୋଡ୍ ସେଟ୍ କରନ୍ତୁ",
    loading: "ଲୋଡ୍ ହେଉଛି...",
    noAlerts: "କୌଣସି ସତର୍କତା ନାହିଁ",
    offline: "ଅଫଲାଇନ",
    noData: "କୌଣସି ଡାଟା ନାହିଁ",
    noReminders: "କୌଣସି ସ୍ମାରକ ନାହିଁ",
  },
  as: {
    greeting: "🙏 নমস্কাৰ",
    idleHint: "কওক বা ফটো লওক",
    voiceLabel: "কওক",
    cameraLabel: "চাওক",
    uploadLabel: "আপল’ড",
    injectBtnLabel: "ডাটা",
    listeningLabel: "শুনি আছে...",
    processingLabel: "বুজি আছে...",
    injectLabel: "জ্ঞান ভঁৰাল",
    injectFileText: "ফাইল বাছক",
    injectSuccess: "✅ ডাটা যোগ হʼল!",
    injectFail: "❌ কিবা সমস্যা হʼল",
    injectProcessing: "⏳ ডাটা পঢ়ি আছে...",
    resultPlaying: "🔊 শুনা গৈ আছে...",
    copied: "✅ কপি হʼল!",
    stageNames: { asr: "🗣 শুনা", nmt: "🔄 অনুবাদ", llm: "🧠 চিন্তা", tts: "🔊 কোৱা" },
    sosLabel: "জৰুৰী",
    weatherLabel: "বতৰ",
    mandiLabel: "মাণ্ডি",
    remindersLabel: "সোঁৱৰণী",
    locationTitle: "আপোনাৰ স্থান নিৰ্বাচন কৰক",
    pincodePlaceholder: "পিন ক'ড লিখক (যেনে 226001)",
    schemesTitle: "আপোনাৰ অঞ্চলৰ আঁচনি:",
    sosHeader: "🚨 জৰুৰী",
    weatherHeader: "🌤 বতৰ সতৰ্কতা",
    mandiHeader: "🌾 মাণ্ডি দাম",
    remindersHeader: "📅 আঁচনি শেষ তাৰিখ",
    setLocationFirst: "📍 আগতে স্থান নিৰ্বাচন কৰক",
    clickPincode: "📍 পিন ক'ড ছেট কৰক",
    loading: "ল'ড হৈ আছে...",
    noAlerts: "কোনো সতৰ্কতা নাই",
    offline: "অফলাইন",
    noData: "কোনো তথ্য নাই",
    noReminders: "কোনো সোঁৱৰণী নাই",
  },
  ur: {
    greeting: "🙏 السلام علیکم",
    idleHint: "بولیں یا تصویر لیں",
    voiceLabel: "بولیں",
    cameraLabel: "دیکھیں",
    uploadLabel: "اپ لوڈ",
    injectBtnLabel: "ڈیٹا",
    listeningLabel: "سن رہا ہے...",
    processingLabel: "سمجھ رہا ہے...",
    injectLabel: "علم کا ذخیرہ",
    injectFileText: "فائل چنیں",
    injectSuccess: "✅ ڈیٹا شامل!",
    injectFail: "❌ کچھ غلط ہوا",
    injectProcessing: "⏳ ڈیٹا پڑھ رہا...",
    resultPlaying: "🔊 سنائی دے رہا...",
    copied: "✅ کاپی ہوا!",
    stageNames: { asr: "🗣 سننا", nmt: "🔄 ترجمہ", llm: "🧠 سوچنا", tts: "🔊 بولنا" },
    sosLabel: "ایمرجنسی",
    weatherLabel: "موسم",
    mandiLabel: "منڈی",
    remindersLabel: "یاد",
    locationTitle: "اپنا مقام منتخب کریں",
    pincodePlaceholder: "پن کوڈ درج کریں (مثلاً 226001)",
    schemesTitle: "آپ کے علاقے کی اسکیمیں:",
    sosHeader: "🚨 ایمرجنسی",
    weatherHeader: "🌤 موسم کی وارننگ",
    mandiHeader: "🌾 منڈی بھاؤ",
    remindersHeader: "📅 اسکیم آخری تاریخ",
    setLocationFirst: "📍 پہلے مقام منتخب کریں",
    clickPincode: "📍 پن کوڈ سیٹ کریں",
    loading: "لوڈ ہو رہا...",
    noAlerts: "کوئی وارننگ نہیں",
    offline: "آف لائن",
    noData: "کوئی ڈیٹا نہیں",
    noReminders: "کوئی یاد نہیں",
  },
  sa: {
    greeting: "🙏 नमस्ते",
    idleHint: "वदतु अथवा छायाचित्रं गृह्णातु",
    voiceLabel: "वदतु",
    cameraLabel: "पश्यतु",
    uploadLabel: "आरोपयतु",
    injectBtnLabel: "दत्तांश",
    listeningLabel: "शृणोति...",
    processingLabel: "अवगच्छति...",
    injectLabel: "ज्ञानकोषः",
    injectFileText: "सञ्चिकां चिनोतु",
    injectSuccess: "✅ दत्तांशः योजितः!",
    injectFail: "❌ किमपि दोषः",
    injectProcessing: "⏳ दत्तांशं पठति...",
    resultPlaying: "🔊 श्रूयते...",
    copied: "✅ प्रतिलिपिता!",
    stageNames: { asr: "🗣 श्रवणम्", nmt: "🔄 अनुवादः", llm: "🧠 चिन्तनम्", tts: "🔊 वचनम्" },
    sosLabel: "आपत्कालः",
    weatherLabel: "वातावरणम्",
    mandiLabel: "विपणिः",
    remindersLabel: "स्मारकम्",
    locationTitle: "स्वस्थानं चिनोतु",
    pincodePlaceholder: "पिन-संकेतं लिखतु (यथा 226001)",
    schemesTitle: "भवतः क्षेत्रस्य योजनाः:",
    sosHeader: "🚨 आपत्कालः",
    weatherHeader: "🌤 वातावरण-सूचना",
    mandiHeader: "🌾 विपणि-मूल्यम्",
    remindersHeader: "📅 योजना-अन्तिमतिथिः",
    setLocationFirst: "📍 प्रथमं स्थानं चिनोतु",
    clickPincode: "📍 पिन-संकेतं स्थापयतु",
    loading: "आह्रियते...",
    noAlerts: "सूचना नास्ति",
    offline: "असम्बद्धम्",
    noData: "दत्तांशः नास्ति",
    noReminders: "स्मारकं नास्ति",
  },
  kok: {
    greeting: "🙏 नमस्कार",
    idleHint: "उलया वा फोटो काडा",
    voiceLabel: "उलया",
    cameraLabel: "पळया",
    uploadLabel: "अपलोड",
    injectBtnLabel: "डेटा",
    listeningLabel: "आयकता...",
    processingLabel: "समजता...",
    injectLabel: "ज्ञान भांडार",
    injectFileText: "फायल निवडा",
    injectSuccess: "✅ डेटा घाला!",
    injectFail: "❌ किदें तरी चुकलें",
    injectProcessing: "⏳ डेटा वाचता...",
    resultPlaying: "🔊 आयकता...",
    copied: "✅ कॉपी जालें!",
    stageNames: { asr: "🗣 आयकप", nmt: "🔄 अणकार", llm: "🧠 विचार", tts: "🔊 उलोवप" },
    sosLabel: "आणीबाणी",
    weatherLabel: "हवामान",
    mandiLabel: "बाजार",
    remindersLabel: "याद",
    locationTitle: "तुमचें स्थान निवडात",
    pincodePlaceholder: "पिन कोड घालात (उदा. 226001)",
    schemesTitle: "तुमच्या भागांतल्यो योजना:",
    sosHeader: "🚨 आणीबाणी",
    weatherHeader: "🌤 हवामान इशारो",
    mandiHeader: "🌾 बाजार भाव",
    remindersHeader: "📅 योजना अंतिम तारीख",
    setLocationFirst: "📍 पयलीं स्थान निवडात",
    clickPincode: "📍 पिन कोड सेट करात",
    loading: "लोड जाता...",
    noAlerts: "इशारे नात",
    offline: "ऑफलाइन",
    noData: "डेटा ना",
    noReminders: "याद ना",
  },
  ks: {
    greeting: "🙏 آداب",
    idleHint: "بولِو یا فوٹو کھینچِو",
    voiceLabel: "بولِو",
    cameraLabel: "وُچھِو",
    uploadLabel: "اَپلوڈ",
    injectBtnLabel: "ڈیٹا",
    listeningLabel: "بوٗزان...",
    processingLabel: "سمجھان...",
    injectLabel: "عِلم خزانہٕ",
    injectFileText: "فایل چُنِو",
    injectSuccess: "✅ ڈیٹا ذٲیع!",
    injectFail: "❌ کانہہ غلطی",
    injectProcessing: "⏳ ڈیٹا پرٛان...",
    resultPlaying: "🔊 بوٗزنَوُن...",
    copied: "✅ کاپی بٲیی!",
    stageNames: { asr: "🗣 بوٗزُن", nmt: "🔄 ترجمہٕ", llm: "🧠 سوچ", tts: "🔊 وَنُن" },
    sosLabel: "ایمرجنسی",
    weatherLabel: "موسم",
    mandiLabel: "منڈی",
    remindersLabel: "یاد",
    locationTitle: "پنُن جاے مُقَرر کرِو",
    pincodePlaceholder: "پِن کوڈ ٹایپ کرِو (مثلاً 226001)",
    schemesTitle: "تُہنٛدِ علاقہٕ ہٕنٛد سکیم:",
    sosHeader: "🚨 ایمرجنسی",
    weatherHeader: "🌤 موسم اطلاع",
    mandiHeader: "🌾 منڈی بھاوٕ",
    remindersHeader: "📅 سکیم آخری تاریخ",
    setLocationFirst: "📍 پہلہٕ جاے مُقَرر کرِو",
    clickPincode: "📍 پِن کوڈ سیٹ کرِو",
    loading: "لوڈ بنان...",
    noAlerts: "کانہہ اطلاع نٲ",
    offline: "آف لایٔن",
    noData: "کانہہ ڈیٹا نٲ",
    noReminders: "کانہہ یاد نٲ",
  },
  mai: {
    greeting: "🙏 प्रणाम",
    idleHint: "बोलू या फोटो लिअ",
    voiceLabel: "बोलू",
    cameraLabel: "देखू",
    uploadLabel: "अपलोड",
    injectBtnLabel: "डेटा",
    listeningLabel: "सुनि रहल...",
    processingLabel: "बुझि रहल...",
    injectLabel: "ज्ञान भंडार",
    injectFileText: "फाइल चुनू",
    injectSuccess: "✅ डेटा जोड़ल!",
    injectFail: "❌ किछु गड़बड़",
    injectProcessing: "⏳ डेटा पढ़ि रहल...",
    resultPlaying: "🔊 सुनाइ दे रहल...",
    copied: "✅ कॉपी भेल!",
    stageNames: { asr: "🗣 सुनब", nmt: "🔄 अनुवाद", llm: "🧠 सोचब", tts: "🔊 बजाब" },
    sosLabel: "आपातकाल",
    weatherLabel: "मौसम",
    mandiLabel: "मंडी",
    remindersLabel: "याद",
    locationTitle: "अपन स्थान चुनू",
    pincodePlaceholder: "पिन कोड दर्ज करू (जेना 226001)",
    schemesTitle: "अहाँक क्षेत्रक योजना:",
    sosHeader: "🚨 आपातकाल",
    weatherHeader: "🌤 मौसम चेतावनी",
    mandiHeader: "🌾 मंडी भाव",
    remindersHeader: "📅 योजना अंतिम तिथि",
    setLocationFirst: "📍 पहिने स्थान चुनू",
    clickPincode: "📍 पिन कोड सेट करू",
    loading: "लोड भ रहल...",
    noAlerts: "कोनो चेतावनी नहि",
    offline: "ऑफलाइन",
    noData: "कोनो डेटा नहि",
    noReminders: "कोनो याद नहि",
  },
  mni: {
    greeting: "🙏 খুরুমজরি",
    idleHint: "ঙাংবিয়ু নত্রগা মখোল লৌবিয়ু",
    voiceLabel: "ঙাংবু",
    cameraLabel: "য়েংবু",
    uploadLabel: "আপলোড",
    injectBtnLabel: "ডাটা",
    listeningLabel: "তারকই...",
    processingLabel: "খংবদোকই...",
    injectLabel: "শিংনবা মফম",
    injectFileText: "ফাইল খনবিয়ু",
    injectSuccess: "✅ ডাটা শিনবিরে!",
    injectFail: "❌ অমত্তা লৈবাক অমা",
    injectProcessing: "⏳ ডাটা পাবদোকই...",
    resultPlaying: "🔊 তাবদোকই...",
    copied: "✅ কপি তৌরে!",
    stageNames: { asr: "🗣 তারকপা", nmt: "🔄 হোৎনবা", llm: "🧠 খনবা", tts: "🔊 ঙাংবা" },
    sosLabel: "ইমর্জেন্সি",
    weatherLabel: "নুংশিৎ",
    mandiLabel: "মান্দি",
    remindersLabel: "নিংশিংবা",
    locationTitle: "নহাক্কী মফম খনবিয়ু",
    pincodePlaceholder: "পিন কোড থাবিয়ু (মখোয় 226001)",
    schemesTitle: "নহাক্কী মফমগী স্কিম:",
    sosHeader: "🚨 ইমর্জেন্সি",
    weatherHeader: "🌤 নুংশিৎ খংহনবা",
    mandiHeader: "🌾 মান্দি মনল",
    remindersHeader: "📅 স্কিম অরোইবা তাং",
    setLocationFirst: "📍 হান্না মফম খনবিয়ু",
    clickPincode: "📍 পিন কোড থমবিয়ু",
    loading: "লোদ তৌই...",
    noAlerts: "খংহনবা লৈতে",
    offline: "অফলাইন",
    noData: "দাতা লৈতে",
    noReminders: "নিংশিংবা লৈতে",
  },
  ne: {
    greeting: "🙏 नमस्ते",
    idleHint: "बोल्नुहोस् वा फोटो लिनुहोस्",
    voiceLabel: "बोल्नु",
    cameraLabel: "हेर्नु",
    uploadLabel: "अपलोड",
    injectBtnLabel: "डेटा",
    listeningLabel: "सुन्दैछ...",
    processingLabel: "बुझ्दैछ...",
    injectLabel: "ज्ञान भण्डार",
    injectFileText: "फाइल छान्नुहोस्",
    injectSuccess: "✅ डेटा थपियो!",
    injectFail: "❌ केही गलत भयो",
    injectProcessing: "⏳ डेटा पढ्दैछ...",
    resultPlaying: "🔊 सुनिँदैछ...",
    copied: "✅ कपि भयो!",
    stageNames: { asr: "🗣 सुन्नु", nmt: "🔄 अनुवाद", llm: "🧠 सोच्नु", tts: "🔊 बोल्नु" },
    sosLabel: "आपतकाल",
    weatherLabel: "मौसम",
    mandiLabel: "मण्डी",
    remindersLabel: "सम्झना",
    locationTitle: "आफ्नो स्थान छान्नुहोस्",
    pincodePlaceholder: "पिन कोड हाल्नुहोस् (जस्तै 226001)",
    schemesTitle: "तपाईंको क्षेत्रको योजना:",
    sosHeader: "🚨 आपतकाल",
    weatherHeader: "🌤 मौसम चेतावनी",
    mandiHeader: "🌾 मण्डी भाउ",
    remindersHeader: "📅 योजना अन्तिम मिति",
    setLocationFirst: "📍 पहिले स्थान छान्नुहोस्",
    clickPincode: "📍 पिन कोड सेट गर्नुहोस्",
    loading: "लोड हुँदैछ...",
    noAlerts: "कुनै चेतावनी छैन",
    offline: "अफलाइन",
    noData: "कुनै डेटा छैन",
    noReminders: "कुनै सम्झना छैन",
  },
  sd: {
    greeting: "🙏 سلام",
    idleHint: "ڳالهايو يا تصوير وٺو",
    voiceLabel: "ڳالهايو",
    cameraLabel: "ڏسو",
    uploadLabel: "اپلوڊ",
    injectBtnLabel: "ڊيٽا",
    listeningLabel: "ٻُڌي رهيو...",
    processingLabel: "سمجهي رهيو...",
    injectLabel: "علم جو خزانو",
    injectFileText: "فائل چونو",
    injectSuccess: "✅ ڊيٽا شامل!",
    injectFail: "❌ ڪجهه غلط ٿيو",
    injectProcessing: "⏳ ڊيٽا پڙهي رهيو...",
    resultPlaying: "🔊 ٻُڌائي رهيو...",
    copied: "✅ ڪاپي ٿيو!",
    stageNames: { asr: "🗣 ٻُڌڻ", nmt: "🔄 ترجمو", llm: "🧠 سوچ", tts: "🔊 ڳالهائڻ" },
    sosLabel: "ايمرجنسي",
    weatherLabel: "موسم",
    mandiLabel: "مَنڊِي",
    remindersLabel: "ياد",
    locationTitle: "پنهنجي جاءِ چونو",
    pincodePlaceholder: "پن ڪوڊ لکو (مثال 226001)",
    schemesTitle: "توهان جي علائقي جون اسڪيمون:",
    sosHeader: "🚨 ايمرجنسي",
    weatherHeader: "🌤 موسم خبردار",
    mandiHeader: "🌾 مَنڊِي ڀاءُ",
    remindersHeader: "📅 اسڪيم آخري تاريخ",
    setLocationFirst: "📍 اول جاءِ چونو",
    clickPincode: "📍 پن ڪوڊ سيٽ ڪيو",
    loading: "لوڊ ٿي رهيو...",
    noAlerts: "ڪا خبردار ڪونهي",
    offline: "آف لائن",
    noData: "ڪو ڊيٽا ڪونهي",
    noReminders: "ڪا ياد ڪونهي",
  },
  doi: {
    greeting: "🙏 नमस्ते जी",
    idleHint: "बोलो जां फोटो खिचो",
    voiceLabel: "बोलो",
    cameraLabel: "देखो",
    uploadLabel: "अपलोड",
    injectBtnLabel: "डेटा",
    listeningLabel: "सुनी रेआ...",
    processingLabel: "समझी रेआ...",
    injectLabel: "ज्ञान भंडार",
    injectFileText: "फाइल चुनो",
    injectSuccess: "✅ डेटा जोड़ी देआ!",
    injectFail: "❌ कुश गड़बड़ होई गेई",
    injectProcessing: "⏳ डेटा पढ़ी रेआ...",
    resultPlaying: "🔊 सुनाई देई रेआ...",
    copied: "✅ कॉपी होई गेआ!",
    stageNames: { asr: "🗣 सुनना", nmt: "🔄 अनुवाद", llm: "🧠 सोचना", tts: "🔊 बोलना" },
    sosLabel: "आपातकाल",
    weatherLabel: "मौसम",
    mandiLabel: "मंडी",
    remindersLabel: "याद",
    locationTitle: "अपनी जगह चुनो",
    pincodePlaceholder: "पिन कोड लिखो (जिय्यां 226001)",
    schemesTitle: "तुंदे इलाके दियां योजनाएं:",
    sosHeader: "🚨 आपातकाल",
    weatherHeader: "🌤 मौसम चेतावनी",
    mandiHeader: "🌾 मंडी भाव",
    remindersHeader: "📅 योजना आखरी तारीख",
    setLocationFirst: "📍 पहलें जगह चुनो",
    clickPincode: "📍 पिन कोड सेट करो",
    loading: "लोड होई रेआ...",
    noAlerts: "कोई चेतावनी नेईं",
    offline: "ऑफलाइन",
    noData: "कोई डेटा नेईं",
    noReminders: "कोई याद नेईं",
  },
  brx: {
    greeting: "🙏 फैसालि",
    idleHint: "रायज्‍लाय नाथाय फोटो लानाय",
    voiceLabel: "रायज्",
    cameraLabel: "नाय",
    uploadLabel: "अपलोड",
    injectBtnLabel: "डाटा",
    listeningLabel: "खोनादों...",
    processingLabel: "मिथिदों...",
    injectLabel: "सिख्ला गोदान",
    injectFileText: "फाइल सायख",
    injectSuccess: "✅ डाटा सोलायनाय!",
    injectFail: "❌ गोनांथार जादों",
    injectProcessing: "⏳ डाटा फरायदों...",
    resultPlaying: "🔊 खोनायदों...",
    copied: "✅ कपि जानाय!",
    stageNames: { asr: "🗣 खोनानाय", nmt: "🔄 रोखा", llm: "🧠 सानग्रा", tts: "🔊 बुंनाय" },
    sosLabel: "गोनांथार",
    weatherLabel: "हावा",
    mandiLabel: "बजार",
    remindersLabel: "सोमजि",
    locationTitle: "नोंथांनि जायगा सायख",
    pincodePlaceholder: "पिन कोड हाबहो (मोनसे 226001)",
    schemesTitle: "नोंथांनि गामिनि स्किम:",
    sosHeader: "🚨 गोनांथार",
    weatherHeader: "🌤 हावा सिनायथि",
    mandiHeader: "🌾 बजार मोन",
    remindersHeader: "📅 स्किम जोबथा सान",
    setLocationFirst: "📍 सिगां जायगा सायख",
    clickPincode: "📍 पिन कोड थानाय",
    loading: "लोड जादों...",
    noAlerts: "सिनायथि गैया",
    offline: "अफलाइन",
    noData: "डाटा गैया",
    noReminders: "सोमजि गैया",
  },
  sat: {
    greeting: "🙏 ᱡᱚᱦᱟᱨ",
    idleHint: "ᱨᱚᱲᱢᱮ ᱵᱟᱝ ᱯᱷᱚᱴᱚ ᱚᱞᱢᱮ",
    voiceLabel: "ᱨᱚᱲ",
    cameraLabel: "ᱧᱮᱞ",
    uploadLabel: "ᱚᱯᱞᱚᱰ",
    injectBtnLabel: "ᱰᱟᱴᱟ",
    listeningLabel: "ᱟᱸᱡᱩᱢᱮᱫ ᱠᱟᱱᱟ...",
    processingLabel: "ᱵᱩᱡᱷᱟᱹᱣᱮᱫ ᱠᱟᱱᱟ...",
    injectLabel: "ᱜᱮᱭᱟᱱ ᱜᱚᱫᱟᱢ",
    injectFileText: "ᱯᱷᱟᱭᱤᱞ ᱵᱟᱪᱷᱟᱣᱢᱮ",
    injectSuccess: "✅ ᱰᱟᱴᱟ ᱥᱮᱞᱮᱫ ᱮᱱᱟ!",
    injectFail: "❌ ᱡᱟᱦᱟᱸᱱ ᱵᱷᱩᱞ ᱦᱩᱭ ᱮᱱᱟ",
    injectProcessing: "⏳ ᱰᱟᱴᱟ ᱯᱟᱲᱷᱟᱣᱮᱫ ᱠᱟᱱᱟ...",
    resultPlaying: "🔊 ᱟᱸᱡᱩᱢᱚᱜ ᱠᱟᱱᱟ...",
    copied: "✅ ᱠᱚᱯᱤ ᱦᱩᱭ ᱮᱱᱟ!",
    stageNames: { asr: "🗣 ᱟᱸᱡᱩᱢ", nmt: "🔄 ᱛᱚᱨᱡᱚᱢᱟ", llm: "🧠 ᱩᱱᱩᱫᱩᱜ", tts: "🔊 ᱨᱚᱲ" },
    sosLabel: "ᱡᱚᱨᱩᱨᱤ",
    weatherLabel: "ᱦᱚᱭᱟ",
    mandiLabel: "ᱢᱟᱸᱰᱤ",
    remindersLabel: "ᱩᱱᱩᱫᱩᱜ",
    locationTitle: "ᱟᱢᱟᱜ ᱡᱟᱭᱜᱟ ᱵᱟᱪᱷᱟᱣᱢᱮ",
    pincodePlaceholder: "ᱯᱤᱱ ᱠᱚᱰ ᱚᱞᱢᱮ (ᱡᱮᱢᱚᱱ 226001)",
    schemesTitle: "ᱟᱢᱟᱜ ᱡᱟᱭᱜᱟ ᱨᱮᱱᱟᱜ ᱥᱠᱤᱢ:",
    sosHeader: "🚨 ᱡᱚᱨᱩᱨᱤ",
    weatherHeader: "🌤 ᱦᱚᱭᱟ ᱪᱮᱛᱟᱣᱱᱤ",
    mandiHeader: "🌾 ᱢᱟᱸᱰᱤ ᱢᱚᱞ",
    remindersHeader: "📅 ᱥᱠᱤᱢ ᱛᱟᱨᱤᱠ",
    setLocationFirst: "📍 ᱥᱤᱜᱟᱹᱧ ᱡᱟᱭᱜᱟ ᱵᱟᱪᱷᱟᱣᱢᱮ",
    clickPincode: "📍 ᱯᱤᱱ ᱠᱚᱰ ᱥᱮᱴ ᱢᱮ",
    loading: "ᱞᱚᱰ ᱦᱩᱭᱩᱜ ᱠᱟᱱᱟ...",
    noAlerts: "ᱡᱟᱦᱟᱸᱱ ᱪᱮᱛᱟᱣᱱᱤ ᱵᱟᱹᱱᱩᱜ",
    offline: "ᱚᱯᱷᱞᱟᱭᱤᱱ",
    noData: "ᱡᱟᱦᱟᱸᱱ ᱰᱟᱴᱟ ᱵᱟᱹᱱᱩᱜ",
    noReminders: "ᱡᱟᱦᱟᱸᱱ ᱩᱱᱩᱫᱩᱜ ᱵᱟᱹᱱᱩᱜ",
  },
};

// ── Translation helper ──
function t(key) {
  return (T[currentUILang] && T[currentUILang][key]) || T.hi[key] || T.en[key] || key;
}

function applyTranslations() {
  const keys = Object.keys(T.hi).filter(k => k !== "stageNames");
  for (const key of keys) {
    const el = document.getElementById(key);
    if (el) el.textContent = t(key);
  }

  const langName = LANG_NAMES[currentUILang] || currentUILang;
  const badge = document.getElementById("langBadge");
  if (badge) badge.textContent = "🌐 " + langName;
  const lisBadge = document.getElementById("listeningLangBadge");
  if (lisBadge) lisBadge.textContent = "🌐 " + langName;

  const pincodeEl = document.getElementById("pincodeInput");
  if (pincodeEl) pincodeEl.placeholder = t("pincodePlaceholder");

  const weatherLoc = document.getElementById("weatherLocation");
  if (weatherLoc && !localStorage.getItem("netra_state")) {
    weatherLoc.textContent = t("setLocationFirst");
  }
  const mandiLoc = document.getElementById("mandiLocation");
  if (mandiLoc && !localStorage.getItem("netra_state")) {
    mandiLoc.textContent = t("setLocationFirst");
  }
}

// ── State Management ─────────────────────────────────
function showState(stateId) {
  if (stateId !== "stateCamera" && cameraStream) {
    cameraStream.getTracks().forEach(t => t.stop());
    cameraStream = null;
  }
  document.querySelectorAll(".state").forEach(s => s.classList.add("hidden"));
  const el = document.getElementById(stateId);
  if (el) el.classList.remove("hidden");
  if (stateId !== "stateLocation") {
    document.getElementById("btnLocation").classList.remove("active");
  }
}

// ── Voice Button ─────────────────────────────────────
document.getElementById("btnVoice").addEventListener("click", async () => {
  if (isRecording) {
    stopRecording();
    return;
  }
  startRecording();
});

async function startRecording() {
  if (isRecording) return;
  isRecording = true;
  detectedSpokenLang = null;
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mimeType = MediaRecorder.isTypeSupported("audio/webm") ? "audio/webm" :
                     MediaRecorder.isTypeSupported("audio/mp4") ? "audio/mp4" : "";
    const recorderOptions = mimeType ? { mimeType } : {};
    recordingMimeType = mimeType || "audio/webm";
    mediaRecorder = new MediaRecorder(stream, recorderOptions);
    audioChunks = [];
    recordingStartTime = Date.now();

    mediaRecorder.ondataavailable = (e) => { if (e.data.size > 0) audioChunks.push(e.data); };
    mediaRecorder.onstop = () => {
      stream.getTracks().forEach(t => t.stop());
      const duration = Date.now() - recordingStartTime;
      if (duration < 1500 || audioChunks.length === 0) {
        showState("stateIdle");
        return;
      }
      processAudio();
    };

    mediaRecorder.start();
    showState("stateListening");
    document.getElementById("btnVoice").classList.add("active");
  } catch (err) {
    isRecording = false;
    console.error("Mic error:", err);
  }
}

function stopRecording() {
  if (mediaRecorder && mediaRecorder.state === "recording") {
    mediaRecorder.stop();
  }
  isRecording = false;
  document.getElementById("btnVoice").classList.remove("active");
}

let pendingAudioBlob = null;

async function processAudio() {
  showState("stateProcessing");
  activatePipelineDot(0);

  const blob = new Blob(audioChunks, { type: recordingMimeType });
  pendingAudioBlob = blob;

  const ext = recordingMimeType === "audio/mp4" ? "mp4" : "webm";
  const formData = new FormData();
  formData.append("audio", blob, `recording.${ext}`);
  formData.append("lang", currentUILang);

  try {
    const res = await fetch(`${API}/api/transcribe`, { method: "POST", body: formData });
    if (!res.ok) {
      const errData = await res.json().catch(() => ({ error: `Server error ${res.status}` }));
      displayResult({ error: true, translation: errData.error || `Server error ${res.status}` });
      return;
    }
    const data = await res.json();

    if (data.error || !data.transcript) {
      displayResult({ error: true, translation: data.error || "Could not understand audio" });
      return;
    }

    if (data.voice_command) {
      handleVoiceCommand(data.voice_command);
      return;
    }

    detectedSpokenLang = data.language || currentUILang;
    showVerifyScreen(data.transcript);
  } catch (err) {
    displayResult({ error: true, translation: "Error: " + err.message });
  }
}

function showVerifyScreen(transcript) {
  document.getElementById("verifyTitle").textContent = t("verifyTitle") || "आपने कहा:";
  document.getElementById("verifyText").textContent = transcript;
  document.getElementById("verifyEditInput").classList.add("hidden");
  document.getElementById("verifyEditInput").value = transcript;
  showState("stateVerify");
}

function getUserLocation() {
  return {
    state: localStorage.getItem("netra_state") || "",
    district: localStorage.getItem("netra_district") || "",
  };
}

async function processConfirmedText(text) {
  showState("stateProcessing");
  activatePipelineDot(1);

  const langToUse = detectedSpokenLang || currentUILang;
  const loc = getUserLocation();

  try {
    const res = await fetch(`${API}/api/pipeline/text`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: text,
        language: langToUse,
        mode: "auto",
        state: loc.state,
        district: loc.district,
      }),
    });
    if (!res.ok) {
      const errData = await res.json().catch(() => ({ error: `Server error ${res.status}` }));
      displayResult({ error: true, translation: errData.error || `Server error ${res.status}` });
      return;
    }
    const data = await res.json();
    data.asr_transcript = text;
    displayResult(data);
  } catch (err) {
    displayResult({ error: true, translation: "Error: " + err.message });
  }
}

document.getElementById("btnConfirm").addEventListener("click", () => {
  const editInput = document.getElementById("verifyEditInput");
  const text = editInput.classList.contains("hidden")
    ? document.getElementById("verifyText").textContent
    : editInput.value;
  processConfirmedText(text);
});

document.getElementById("btnEdit").addEventListener("click", () => {
  const editInput = document.getElementById("verifyEditInput");
  if (editInput.classList.contains("hidden")) {
    editInput.classList.remove("hidden");
    editInput.focus();
  } else {
    editInput.classList.add("hidden");
  }
});

document.getElementById("btnRetry").addEventListener("click", () => {
  showState("stateIdle");
  pendingAudioBlob = null;
});

document.getElementById("verifyEditInput").addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    processConfirmedText(e.target.value);
  }
});

// ── Camera Button ────────────────────────────────────
document.getElementById("btnCamera").addEventListener("click", async () => {
  if (cameraStream) {
    stopCamera();
    return;
  }
  startCamera();
});

async function startCamera() {
  const constraintsList = [
    { video: { facingMode: { ideal: "environment" }, width: { ideal: 640 }, height: { ideal: 480 } } },
    { video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } } },
    { video: true },
  ];

  let lastErr = null;
  for (const constraints of constraintsList) {
    try {
      cameraStream = await navigator.mediaDevices.getUserMedia(constraints);
      break;
    } catch (err) {
      lastErr = err;
      cameraStream = null;
    }
  }

  if (!cameraStream) {
    console.error("Camera error:", lastErr);
    const msg = (lastErr && lastErr.name === "NotAllowedError")
      ? "Camera permission denied. Allow camera access and try again."
      : "Camera not available on this device. Check webcam connection.";
    displayResult({ error: true, translation: msg });
    return;
  }

  try {
    const video = document.getElementById("cameraPreview");
    video.srcObject = cameraStream;
    video.muted = true;
    video.setAttribute("playsinline", "true");
    await video.play().catch(() => {});
    showState("stateCamera");
  } catch (err) {
    console.error("Camera preview error:", err);
    stopCamera();
    displayResult({ error: true, translation: "Could not start camera preview." });
  }
}

function stopCamera() {
  if (cameraStream) {
    cameraStream.getTracks().forEach(t => t.stop());
    cameraStream = null;
  }
  const video = document.getElementById("cameraPreview");
  if (video) video.srcObject = null;
  showState("stateIdle");
}

document.getElementById("btnCapture").addEventListener("click", () => {
  const video = document.getElementById("cameraPreview");
  const canvas = document.getElementById("captureCanvas");

  if (!video || !video.videoWidth || !video.videoHeight) {
    displayResult({ error: true, translation: "Camera not ready. Wait for preview, then capture." });
    return;
  }

  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext("2d").drawImage(video, 0, 0);

  // Keep a copy of the frame before stopping the stream
  stopCamera();
  showState("stateProcessing");
  activatePipelineDot(0);

  canvas.toBlob(async (blob) => {
    if (!blob || blob.size < 100) {
      displayResult({ error: true, translation: "Capture failed — empty image. Try again." });
      return;
    }

    const formData = new FormData();
    const loc = getUserLocation();
    formData.append("image", blob, "capture.jpg");
    formData.append("lang", currentUILang);
    formData.append("language", currentUILang);
    formData.append("mode", "camera");
    formData.append("state", loc.state);
    formData.append("district", loc.district);

    try {
      const res = await fetch(`${API}/api/pipeline/run`, { method: "POST", body: formData });
      if (!res.ok) {
        const errData = await res.json().catch(() => ({ error: `Server error ${res.status}` }));
        displayResult({ error: true, translation: errData.error || `Server error ${res.status}` });
        return;
      }
      const data = await res.json();
      displayResult(data);
    } catch (err) {
      displayResult({ error: true, translation: "Error: " + err.message });
    }
  }, "image/jpeg", 0.85);
});

// ── Upload Image / PDF (same pipeline as camera scan) ─
document.getElementById("btnUpload").addEventListener("click", () => {
  if (cameraStream) stopCamera();
  document.getElementById("uploadInput").click();
});

document.getElementById("uploadInput").addEventListener("change", async (e) => {
  const file = e.target.files && e.target.files[0];
  e.target.value = "";
  if (!file) return;

  const name = (file.name || "").toLowerCase();
  const isPdf = file.type === "application/pdf" || name.endsWith(".pdf");
  const isImage = (file.type || "").startsWith("image/") ||
    /\.(png|jpe?g|webp|gif|bmp|tiff?)$/i.test(name);

  if (!isPdf && !isImage) {
    displayResult({ error: true, translation: t("uploadBadType") || "Please upload an image or PDF." });
    return;
  }

  showState("stateProcessing");
  activatePipelineDot(0);

  const formData = new FormData();
  const loc = getUserLocation();
  // Backend accepts image|file; proxy picks field from extension
  formData.append(isPdf ? "file" : "image", file, file.name || (isPdf ? "upload.pdf" : "upload.jpg"));
  formData.append("lang", currentUILang);
  formData.append("language", currentUILang);
  formData.append("mode", "upload");
  formData.append("state", loc.state);
  formData.append("district", loc.district);

  try {
    const res = await fetch(`${API}/api/pipeline/run`, { method: "POST", body: formData });
    if (!res.ok) {
      const errData = await res.json().catch(() => ({ error: `Server error ${res.status}` }));
      displayResult({ error: true, translation: errData.error || `Server error ${res.status}` });
      return;
    }
    const data = await res.json();
    displayResult(data);
  } catch (err) {
    displayResult({ error: true, translation: "Error: " + err.message });
  }
});

// ── Data Inject Button ───────────────────────────────
document.getElementById("btnInject").addEventListener("click", () => {
  showState("stateInject");
  loadKBStats();
});

async function loadKBStats() {
  try {
    const res = await fetch(`${API}/api/ingest/status`);
    const data = await res.json();
    const kb = data.knowledge_base || {};
    const w = data.watcher || {};
    document.getElementById("kbDocsCount").textContent = kb.total_documents || 0;
    document.getElementById("kbChunksCount").textContent = kb.total_chunks || 0;
    const dot = document.getElementById("watcherDot");
    if (w.is_running) {
      dot.classList.remove("inactive");
    } else {
      dot.classList.add("inactive");
    }
    if (w.pending > 0) {
      document.getElementById("injectStatus").textContent = `${w.pending} ${t("filesInInbox")}`;
    }
  } catch (_) {}
}

document.getElementById("injectFileInput").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;

  const status = document.getElementById("injectStatus");
  status.textContent = t("injectProcessing");

  const formData = new FormData();
  const loc = getUserLocation();
  formData.append("file", file);
  formData.append("lang", currentUILang);
  formData.append("language", currentUILang);
  formData.append("state", loc.state);
  formData.append("district", loc.district);

  try {
    const res = await fetch(`${API}/api/ingest`, { method: "POST", body: formData });
    const data = await res.json();
    status.textContent = data.success ? t("injectSuccess") : t("injectFail");
    loadKBStats();
    setTimeout(() => showState("stateIdle"), 2000);
  } catch (err) {
    status.textContent = t("injectFail");
  }
  e.target.value = "";
});

document.getElementById("injectBulkInput").addEventListener("change", async (e) => {
  const files = e.target.files;
  if (!files || files.length === 0) return;

  const status = document.getElementById("injectStatus");
  status.textContent = `${files.length} ${t("bulkUploading")}`;

  const formData = new FormData();
  const loc = getUserLocation();
  for (const file of files) {
    formData.append("files", file);
  }
  formData.append("lang", currentUILang);
  formData.append("language", currentUILang);
  formData.append("state", loc.state);
  formData.append("district", loc.district);

  try {
    const res = await fetch(`${API}/api/ingest/bulk`, { method: "POST", body: formData });
    const data = await res.json();
    status.textContent = t("bulkResult").replace("{s}", data.processed || 0).replace("{f}", data.failed || 0);
    loadKBStats();
    setTimeout(() => showState("stateIdle"), 3000);
  } catch (err) {
    status.textContent = t("injectFail");
  }
  e.target.value = "";
});

document.getElementById("btnScanInbox").addEventListener("click", async () => {
  const status = document.getElementById("injectStatus");
  status.textContent = t("scanInboxText") + "...";

  try {
    const res = await fetch(`${API}/api/ingest/scan`, { method: "POST" });
    const data = await res.json();
    if (data.processed > 0 || data.failed > 0) {
      status.textContent = t("bulkResult").replace("{s}", data.processed || 0).replace("{f}", data.failed || 0);
    } else {
      status.textContent = "✓";
    }
    loadKBStats();
    setTimeout(() => { status.textContent = ""; }, 3000);
  } catch (err) {
    status.textContent = t("injectFail");
  }
});

// ── Pipeline Progress ────────────────────────────────
function activatePipelineDot(idx) {
  const dots = document.querySelectorAll("#pipelineDots .dot");
  dots.forEach((d, i) => {
    d.classList.remove("active", "done");
    if (i < idx) d.classList.add("done");
    if (i === idx) d.classList.add("active");
  });
}

// ── Voice Command Handler ────────────────────────────
let lastResultData = null;

async function handleVoiceCommand(command) {
  switch (command) {
    case "replay":
      const audio = document.getElementById("resultAudio");
      if (hasAudioResult) {
        audio.play();
        showState("stateResult");
      }
      break;
    case "explain_more":
      await sendFollowUp("Isko aur detail mein samjhao. Pura explain karo step by step.");
      break;
    case "action_items":
      await sendFollowUp("Isme kya karna chahiye? Action items batao.");
      break;
    case "deadlines":
      await sendFollowUp("Isme koi deadline ya last date hai? Batao.");
      break;
    case "identify_sender":
      await sendFollowUp("Yeh document kisne bheja hai? Sender kaun hai?");
      break;
    case "change_language":
      switchToNextLanguage();
      break;
    default:
      showState("stateIdle");
  }
}

function switchToNextLanguage() {
  const langSelect = document.getElementById("uiLang");
  const options = Array.from(langSelect.options);
  const currentIdx = options.findIndex(o => o.value === currentUILang);
  const nextIdx = (currentIdx + 1) % options.length;
  currentUILang = options[nextIdx].value;
  localStorage.setItem("netra_lang", currentUILang);
  langSelect.value = currentUILang;
  applyTranslations();
  showState("stateIdle");
}

async function sendFollowUp(followUpText) {
  showState("stateProcessing");
  activatePipelineDot(0);
  const loc = getUserLocation();
  try {
    const res = await fetch(`${API}/api/pipeline/text`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: followUpText,
        language: currentUILang,
        mode: "chat",
        state: loc.state,
        district: loc.district,
      }),
    });
    const data = await res.json();
    displayResult(data);
  } catch (err) {
    displayResult({ error: true, translation: "Error: " + err.message });
  }
}

// ── HTML Escape Helper ──────────────────────────────
function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

// ── Display Result ───────────────────────────────────
function looksLikeLatin(text) {
  if (!text) return false;
  const letters = text.replace(/[^A-Za-z\u0900-\u097F\u0980-\u09FF\u0A00-\u0A7F\u0A80-\u0AFF\u0B00-\u0B7F\u0B80-\u0BFF\u0C00-\u0C7F\u0C80-\u0CFF\u0D00-\u0D7F\u0600-\u06FF]/g, "");
  if (!letters) return false;
  const latin = (letters.match(/[A-Za-z]/g) || []).length;
  return latin / letters.length > 0.55;
}

function displayResult(data) {
  let text = "—";
  let audioUrl = null;
  let questionText = "";

  if (data.voice_command) {
    handleVoiceCommand(data.voice_command);
    return;
  }

  if (data.stages) {
    const nmt = data.stages.nmt;
    const llm = data.stages.llm;
    const ocr = data.stages.ocr;
    const tts = data.stages.tts;

    if (nmt && nmt.success && nmt.data && nmt.data.translated_text) {
      text = nmt.data.translated_text;
    } else if (llm && llm.success && llm.data && llm.data.response) {
      text = llm.data.response;
    } else if (ocr && ocr.success && ocr.data && ocr.data.text) {
      text = ocr.data.text;
    }

    if (tts && tts.success && tts.data && tts.data.audio_path) {
      const fname = tts.data.audio_path.split("/").pop();
      audioUrl = `${API}/api/output/${fname}`;
    }

    // Voice: show what user said. Document OCR stays hidden when UI language
    // differs (e.g. English scan + Hindi UI → only Hindi answer).
    if (data.asr_transcript) {
      const q = String(data.asr_transcript).trim();
      const uiIsIndic = currentUILang && currentUILang !== "en";
      if (!(uiIsIndic && looksLikeLatin(q))) {
        questionText = q;
      }
    }
  } else {
    text = data.translation || data.llm_response || data.ocr_text || data.error || "—";
    if (data.error && !data.stages) text = typeof data.error === "string" ? data.error : "Something went wrong";
    audioUrl = data.audio_url || null;
  }

  lastResultData = data;

  const resultEl = document.getElementById("resultText");
  if (questionText) {
    const qLabel = t("questionLabel") || "Q";
    resultEl.innerHTML = `<div class="question-label">${escapeHtml(qLabel)}: ${escapeHtml(questionText)}</div><div class="answer-text">${escapeHtml(text)}</div>`;
  } else {
    resultEl.textContent = text;
  }
  showState("stateResult");

  if (audioUrl) {
    const audioEl = document.getElementById("resultAudio");
    audioEl.src = audioUrl;
    hasAudioResult = true;
    audioEl.load();
    audioEl.oncanplaythrough = () => {
      audioEl.play().catch(() => {});
    };
    audioEl.onerror = () => {
      hasAudioResult = false;
    };
  } else {
    const audioEl = document.getElementById("resultAudio");
    audioEl.removeAttribute("src");
    hasAudioResult = false;
  }
}

// ── Result Actions ───────────────────────────────────
document.getElementById("btnPlayResult").addEventListener("click", () => {
  const audio = document.getElementById("resultAudio");
  if (hasAudioResult) {
    audio.play().catch(() => {});
  }
});

document.getElementById("btnCopyResult").addEventListener("click", () => {
  const el = document.getElementById("resultText");
  const text = el.innerText || el.textContent;
  navigator.clipboard.writeText(text).catch(() => {});
});

document.getElementById("btnNewQuery").addEventListener("click", () => {
  showState("stateIdle");
});

// ── Language Switcher ────────────────────────────────
document.getElementById("uiLang").addEventListener("change", (e) => {
  currentUILang = e.target.value;
  localStorage.setItem("netra_lang", currentUILang);
  applyTranslations();
  fetchNews();
});

// ── Location Feature ─────────────────────────────────
let savedPincode = localStorage.getItem("netra_pincode") || "";
let savedState = localStorage.getItem("netra_state") || "";

document.getElementById("btnLocation").addEventListener("click", () => {
  showState("stateLocation");
  document.getElementById("btnLocation").classList.add("active");
  if (savedPincode) {
    document.getElementById("pincodeInput").value = savedPincode;
    lookupPincode(savedPincode);
  }
});

document.getElementById("btnPincodeSubmit").addEventListener("click", () => {
  const pin = document.getElementById("pincodeInput").value.trim();
  if (pin.length === 6 && /^\d{6}$/.test(pin)) {
    lookupPincode(pin);
  } else {
    document.getElementById("locationStatus").textContent = "Enter valid 6-digit PIN code";
  }
});

document.getElementById("pincodeInput").addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    document.getElementById("btnPincodeSubmit").click();
  }
});

async function lookupPincode(pincode) {
  const statusEl = document.getElementById("locationStatus");
  statusEl.textContent = "Looking up...";

  try {
    const res = await fetch(`${API}/api/location/lookup`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ pincode }),
    });
    const data = await res.json();

    if (!res.ok) {
      statusEl.textContent = data.error || "PIN code not found";
      return;
    }

    savedPincode = pincode;
    savedState = data.state;
    localStorage.setItem("netra_pincode", pincode);
    localStorage.setItem("netra_state", data.state);
    if (data.district) {
      localStorage.setItem("netra_district", data.district);
    }

    document.getElementById("locationInfo").classList.remove("hidden");
    document.getElementById("locationDetected").textContent = `${data.state} (${data.state_code})`;
    document.getElementById("locationState").textContent = `PIN: ${pincode}`;

    const schemesList = document.getElementById("schemesList");
    schemesList.innerHTML = "";
    const allSchemes = [...data.central_schemes, ...data.state_schemes];
    allSchemes.forEach(s => {
      const li = document.createElement("li");
      li.textContent = s;
      schemesList.appendChild(li);
    });
    document.getElementById("locationSchemes").classList.remove("hidden");
    statusEl.textContent = `${data.central_schemes.length} central + ${data.state_schemes.length} state schemes`;
  } catch (err) {
    statusEl.textContent = "Network error - try again";
  }
}

// ── News Ticker (in idle area) ───────────────────────
let newsLoaded = false;
let newsItems = [];
let newsIndex = 0;
let newsInterval = null;

function initNewsTicker() {
  if (!navigator.onLine) return;
  fetchNews();
  setInterval(() => {
    if (navigator.onLine) fetchNews();
  }, 5 * 60 * 1000);
}

async function fetchNews() {
  try {
    const res = await fetch(`${API}/api/news?lang=${currentUILang || "hi"}`);
    const data = await res.json();
    if (data.news && data.news.length > 0) {
      newsItems = data.news;
      newsLoaded = true;
      startNewsRotation();
    }
  } catch (e) {
    // offline or server unavailable
  }
}

function startNewsRotation() {
  if (newsInterval) clearInterval(newsInterval);
  newsIndex = 0;

  const track = document.getElementById("idleNewsTrack");
  if (track) track.innerHTML = "";

  showNewsItem();
  newsInterval = setInterval(() => {
    newsIndex = (newsIndex + 1) % (newsItems.length + 1);
    showNewsItem();
  }, 4000);
}

function showNewsItem() {
  const track = document.getElementById("idleNewsTrack");
  if (!track) return;

  track.innerHTML = "";

  const item = document.createElement("div");
  item.className = "idle-news-item active";

  if (newsIndex === 0) {
    item.textContent = t("greeting");
  } else {
    const news = newsItems[newsIndex - 1];
    item.textContent = `📰 ${news.title}`;
  }

  track.appendChild(item);
}

window.addEventListener("online", () => {
  if (!newsLoaded) initNewsTicker();
});

window.addEventListener("offline", () => {
  if (newsInterval) {
    clearInterval(newsInterval);
    newsInterval = null;
  }
});

// ── SOS Button ───────────────────────────────────────
document.getElementById("btnSOS").addEventListener("click", () => {
  showState("stateSOS");
});

// ── Weather Button ───────────────────────────────────
document.getElementById("btnWeather").addEventListener("click", () => {
  showState("stateWeather");
  loadWeather();
});

async function loadWeather() {
  const state = localStorage.getItem("netra_state");
  if (!state) {
    document.getElementById("weatherLocation").textContent = t("setLocationFirst");
    document.getElementById("weatherAlerts").innerHTML = `<div class="weather-loading">${t("clickPincode")}</div>`;
    return;
  }
  document.getElementById("weatherLocation").textContent = `📍 ${state}`;
  document.getElementById("weatherAlerts").innerHTML = `<div class="weather-loading">${t("loading")}</div>`;

  try {
    const res = await fetch(`${API}/api/weather?state=${encodeURIComponent(state)}`);
    const data = await res.json();
    const alertsEl = document.getElementById("weatherAlerts");
    if (data.alerts && data.alerts.length > 0) {
      alertsEl.innerHTML = data.alerts.map(a =>
        `<div class="weather-alert-item ${a.type}"><span>${a.icon}</span><span>${a.message}</span></div>`
      ).join("");
    } else {
      alertsEl.innerHTML = `<div class="weather-loading">${t("noAlerts")}</div>`;
    }
  } catch (e) {
    document.getElementById("weatherAlerts").innerHTML = `<div class="weather-loading">${t("offline")}</div>`;
  }
}

// ── Mandi Button ─────────────────────────────────────
document.getElementById("btnMandi").addEventListener("click", () => {
  showState("stateMandi");
  loadMandi();
});

async function loadMandi() {
  const state = localStorage.getItem("netra_state");
  if (!state) {
    document.getElementById("mandiLocation").textContent = t("setLocationFirst");
    document.getElementById("mandiList").innerHTML = `<div class="weather-loading">${t("clickPincode")}</div>`;
    return;
  }
  document.getElementById("mandiLocation").textContent = `📍 ${state}`;
  document.getElementById("mandiList").innerHTML = `<div class="weather-loading">${t("loading")}</div>`;

  try {
    const res = await fetch(`${API}/api/mandi?state=${encodeURIComponent(state)}`);
    const data = await res.json();
    const listEl = document.getElementById("mandiList");
    if (data.prices && data.prices.length > 0) {
      listEl.innerHTML = data.prices.map(p =>
        `<div class="mandi-item"><span class="mandi-crop">${p.crop}</span><span class="mandi-price">${p.price} ${p.unit}</span></div>`
      ).join("");
    } else {
      listEl.innerHTML = `<div class="weather-loading">${t("noData")}</div>`;
    }
  } catch (e) {
    document.getElementById("mandiList").innerHTML = `<div class="weather-loading">${t("offline")}</div>`;
  }
}

// ── Reminders Button ─────────────────────────────────
document.getElementById("btnReminders").addEventListener("click", () => {
  showState("stateReminders");
  loadReminders();
});

async function loadReminders() {
  document.getElementById("remindersList").innerHTML = `<div class="weather-loading">${t("loading")}</div>`;
  try {
    const res = await fetch(`${API}/api/reminders`);
    const data = await res.json();
    const listEl = document.getElementById("remindersList");
    if (data.reminders && data.reminders.length > 0) {
      listEl.innerHTML = data.reminders.map(r =>
        `<div class="reminder-item ${r.urgent ? 'reminder-urgent' : ''}">
          <span class="reminder-icon">${r.icon}</span>
          <div class="reminder-info">
            <div class="reminder-name">${r.name}</div>
            <div class="reminder-date">${r.date} — ${r.desc}</div>
          </div>
        </div>`
      ).join("");
    } else {
      listEl.innerHTML = `<div class="weather-loading">${t("noReminders")}</div>`;
    }
  } catch (e) {
    document.getElementById("remindersList").innerHTML = `<div class="weather-loading">${t("offline")}</div>`;
  }
}

// ── Language Welcome Screen ──────────────────────────
function initLangWelcome() {
  const welcome = document.getElementById("langWelcome");
  const mainApp = document.getElementById("mainApp");

  const track = document.getElementById("idleNewsTrack");
  if (track) track.innerHTML = `<div class="idle-news-item active">${t("greeting")}</div>`;

  if (currentUILang) {
    welcome.classList.add("hidden");
    mainApp.classList.remove("hidden");
    document.getElementById("uiLang").value = currentUILang;
    applyTranslations();
    return;
  }

  welcome.classList.remove("hidden");
  mainApp.classList.add("hidden");

  document.getElementById("langGrid").addEventListener("click", (e) => {
    const tile = e.target.closest(".lang-tile");
    if (!tile) return;

    const lang = tile.dataset.lang;
    currentUILang = lang;
    localStorage.setItem("netra_lang", lang);

    document.getElementById("uiLang").value = lang;
    applyTranslations();

    welcome.style.transition = "opacity 0.3s ease";
    welcome.style.opacity = "0";
    setTimeout(() => {
      welcome.classList.add("hidden");
      welcome.style.opacity = "";
      welcome.style.transition = "";
      mainApp.classList.remove("hidden");
    }, 300);
  });
}

// ── Init ─────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  initLangWelcome();

  fetch(`${API}/api/ready`)
    .then(async (r) => {
      const data = await r.json().catch(() => ({}));
      if (r.ok && data.ready) {
        document.getElementById("statusDot").className = "status-dot green";
        initNewsTicker();
      } else {
        document.getElementById("statusDot").className = "status-dot red";
      }
    })
    .catch(() => {
      document.getElementById("statusDot").className = "status-dot red";
    });
});
