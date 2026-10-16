import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AppProvider } from './context/AppContext';
import { LinkIdentityScreen } from './screens/LinkIdentityScreen';
import { AadhaarScreen } from './screens/AadhaarScreen';
import { OtpScreen } from './screens/OtpScreen';
import { ScoringScreen } from './screens/ScoringScreen';
import { IdentityVerifiedScreen } from './screens/IdentityVerifiedScreen';

export default function App() {
  return (
    <AppProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<LinkIdentityScreen />} />
          <Route path="/aadhaar" element={<AadhaarScreen />} />
          <Route path="/otp" element={<OtpScreen />} />
          <Route path="/scoring" element={<ScoringScreen />} />
          <Route path="/result" element={<IdentityVerifiedScreen />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </BrowserRouter>
    </AppProvider>
  );
}
