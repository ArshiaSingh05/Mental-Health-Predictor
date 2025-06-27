import React from "react"
import { price } from "../../dummydata"

const PriceCard = () => {
  return (
    <>
      {price.map((val) => (
        <div className='items shadow'>
          <h2>{val.name}</h2>
          <h1>
            <span>₹</span>
            {val.price}
          </h1>
          <p>{val.desc}</p>
          {/* Feature List as Bullet Points */}
          <ul className="feature-list">
            {val.features.map((feature, i) => (
              <li key={i}>✔️ {feature}</li>
            ))}
          </ul>
          <button className='outline-btn'>GET STARTED</button>
        </div>
      ))}
    </>
  )
}

export default PriceCard